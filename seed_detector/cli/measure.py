import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from cloup import (
    group,
    command,
    option,
    help_option,
    Path as PathType,
    Choice,
    option_group,
)


def run_single(image_file: Path, mask_file: Path):
    image_name = image_file.with_suffix('').name

    import numpy as np
    import pandas as pd
    import cv2
    import skimage.transform
    import skimage.measure
    import skimage.morphology
    import pyefd
    from ..defaults import DEFAULT_N_POLYGON_VERTICES
    from ..tools import (
        get_contours,
        resample_polygon,
        rotate_upright,
        align_shapes,
        extract_subimage,
        get_bbox,
        polygon_area,
    )

    # read in image and mask
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(image_file))

    # get contours from mask
    n_vertices = DEFAULT_N_POLYGON_VERTICES
    min_size = 1  # we assume only perfect masks
    contour = get_contours(mask, min_size=min_size)[0]
    contour = resample_polygon(contour, n_points=n_vertices)

    props = skimage.measure.regionprops(mask)[0]

    bbox = props.bbox
    length = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    aspect_ratio = length / width

    area = props.area
    perimeter = props.perimeter

    chull = skimage.morphology.convex_hull_image(mask) * 255
    chull_props = skimage.measure.regionprops(chull)[0]
    chull_perimeter = chull_props.perimeter

    diaspore_surface_structure = chull_perimeter / perimeter

    # TODO:
    # dominant color (RGB)
    # dominant color (HSV)
    # dominant color (Lab)
    # Texture

    measurements = pd.DataFrame(
        dict(
            image_name=image_name,
            image_file=str(image_file),
            mask_file=str(mask_file),
            length=length,
            width=width,
            aspect_ratio=aspect_ratio,
            area=area,
            perimeter=perimeter,
            diaspore_surface_structure=diaspore_surface_structure,
        ),
        index=[image_name],
    )

    return measurements


def single_wrapped(args):
    run_single(*args)


@command('single', help='Calculate measurements for a single image-mask pair.')
@option_group(
    'Required options',
    option(
        '-i',
        '--image_file',
        type=PathType(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help='Input image file.',
        required=True,
    ),
    option(
        '-m',
        '--mask_file',
        type=PathType(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help='Input mask file.',
        required=True,
    ),
    option(
        '-o',
        '--out_file',
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f'''
            Output (CSV) file. Will be created if it does not exist. By default
            a file with the name "<IMAGE_FILE>_measurements.csv"
            will be created in the current working directory ({Path(".").resolve()}).
        ''',
        default=None,
        required=False,
    ),
)
@help_option('-h', '--help')
def single(image_file: Path, mask_file: Path, out_file: Optional[Path]):
    image_name = image_file.with_suffix('').name
    if out_file is None:
        out_dir = Path(f'{image_name}_aligned')

    measurements = run_single(image_file=image_file, mask_file=mask_file,)
    print(measurements)


@command('multi', help='Calculate measurements for multiple image-mask pairs.')
@option_group(
    'Required options',
    option(
        '-i',
        '--image_dir',
        type=PathType(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help='Input image directory.',
        required=True,
    ),
    option(
        '-m',
        '--mask_dir',
        type=PathType(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        # TODO:
        help='''
            Input mask directory or align output directory. Mask files must
            be PNG files named <IMAGE_NAME>_mask.png; i.e., for an image file
            "image_1.jpg" the corresponding mask must be named "image_1_mask.png".
        ''',
        required=True,
    ),
    option(
        '-o',
        '--out_file',
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f'''
            Output (CSV) file. By default
            a file with the name "<IMAGE_DIR>_measurements.csv" will be created in
            the current working directory ({Path(".").resolve()}).
        ''',
        default=None,
        required=False,
    ),
)
@option(
    '-t',
    '--n_proc',
    default=1,
    help='''
        Number of parallel processes to run. Should not be set higher than
        the number of CPU cores.
    ''',
)
@help_option('-h', '--help')
def multi(
    image_dir: Path, mask_dir: Path, out_dir: Optional[Path], padding: int, n_proc: int,
):
    # import numpy as np
    # from ..defaults import IMAGE_EXTENSIONS

    # image_dir_name = image_dir.name
    # if out_dir is None:
    #     out_dir = Path(f'{image_dir_name}_aligned')

    # image_extensions = IMAGE_EXTENSIONS
    # image_files = np.array(
    #     [f for f in image_dir.glob('*') if f.suffix.lower() in image_extensions]
    # )
    # image_names = np.array([f.with_suffix('').name for f in image_files])
    pass


@group(
    'measure',
    help='Calculate several different measurements for image-mask pairs.',
    no_args_is_help=True,
)
@help_option('-h', '--help')
def measure():
    pass


measure.add_command(single)
measure.add_command(multi)
