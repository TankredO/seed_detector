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


def run_single(
    image_file: Path,
    mask_file: Path,
    n_colors: int = 5,
    symmetric: bool = True,
    normed: bool = True,
    distances: Iterable[int] = (3, 5),
    angles: Iterable[int] = (0, 45, 90),
    resize_height: Optional[int] = None,
    resize_width: Optional[int] = None,
):
    image_name = image_file.with_suffix('').name

    import numpy as np
    import pandas as pd
    import cv2
    import skimage.transform
    import skimage.measure
    import skimage.morphology
    import skimage.color
    import skimage.feature
    from ..defaults import DEFAULT_N_POLYGON_VERTICES
    from ..tools import (
        get_contours,
        resample_polygon,
        primary_colors,
        resize_image,
    )

    # read in image and mask
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(image_file))

    # get contours from mask
    n_vertices = DEFAULT_N_POLYGON_VERTICES
    min_size = 1  # we assume only perfect masks
    try:
        contour = get_contours(mask, min_size=min_size)[0]
        contour = resample_polygon(contour, n_points=n_vertices)
    except:
        print(f'WARNING: no contour found in {mask_file}')

    # == shape
    props = skimage.measure.regionprops(mask)[0]

    bbox = props.bbox
    length = bbox[2] - bbox[0]
    width = bbox[3] - bbox[1]
    aspect_ratio = length / width

    area = props.area
    perimeter = props.perimeter

    chull = skimage.morphology.convex_hull_image(mask) * 255
    chull_props = skimage.measure.regionprops(chull)[0]
    chull_perimeter = chull_props.perimeter

    diaspore_surface_structure = chull_perimeter / perimeter

    # == dominant colors
    def build_pc_dict(
        colors,
        counts,
        prefix,
        col_comp_names,
    ):
        if np.issubdtype(counts[0], int):
            suffix = 'count'
        else:
            suffix = 'frac'

        pc_dict = {}
        for i, col in enumerate(colors):
            for val, c in zip(col, col_comp_names):
                pc_dict[f'{prefix}_{i}_{c}'] = val
        for i, count in enumerate(counts):
            pc_dict[f'{prefix}_{i}_{suffix}'] = count

        return pc_dict

    # dominant color (RGB)
    colors_rgb, counts_rgb = primary_colors(image[:, :, [2, 1, 0]], mask, n_colors)
    colors_rgb = np.round(colors_rgb, 0).astype(np.uint8)
    frac_rgb = counts_rgb / counts_rgb.sum()
    colors_rgb_dict = build_pc_dict(colors_rgb, frac_rgb, 'rgb', ('r', 'g', 'b'))

    # dominant color (HSV)
    image_hsv = skimage.color.rgb2hsv(image[:, :, [2, 1, 0]])
    colors_hsv, counts_hsv = primary_colors(image_hsv, mask, n_colors)
    frac_hsv = counts_hsv / counts_hsv.sum()
    colors_hsv_dict = build_pc_dict(colors_hsv, frac_hsv, 'hsv', ('h', 's', 'v'))

    # dominant color (Lab)
    image_lab = skimage.color.rgb2lab(image[:, :, [2, 1, 0]])
    colors_lab, counts_lab = primary_colors(image_lab, mask, n_colors)
    frac_lab = counts_lab / counts_lab.sum()
    colors_lab_dict = build_pc_dict(colors_lab, frac_lab, 'lab', ('l', 'a', 'b'))

    # == Texture
    def build_texture_dict(
        props,
        distances,
        angles,
        prefix,
    ):
        texture_dict = {}
        for p_name, p in props.items():
            for i, d in enumerate(distances):
                for j, a in enumerate(angles):
                    texture_dict[f'{prefix}_{p_name}_d{d}_a{a}'] = p[i][j]

        return texture_dict

    p_names = (
        'contrast',
        'dissimilarity',
        'homogeneity',
        'ASM',
        'energy',
        'correlation',
    )

    mask_resized = resize_image(mask, height=resize_height, width=resize_width, order=0)

    # texture gray image
    image_gray = skimage.color.rgb2gray(image[:, :, [2, 1, 0]])
    image_gray = (
        resize_image(image_gray, height=resize_height, width=resize_width) * 255
    ).astype(np.uint8)
    image_gray[mask_resized == 0] = 0

    glcm_gray = skimage.feature.graycomatrix(
        image_gray,
        distances=distances,
        angles=angles,
        symmetric=symmetric,
        normed=normed,
    )
    glcm_gray_props = {
        p_name: skimage.feature.graycoprops(glcm_gray, prop=p_name)
        for p_name in p_names
    }
    glcm_gray_dict = build_texture_dict(glcm_gray_props, distances, angles, 'gray')

    # texture L* (Lab color space)
    image_l = skimage.color.rgb2lab(image[:, :, [2, 1, 0]])[:, :, 0]
    image_l = np.round(
        resize_image(image_l, height=resize_height, width=resize_width)
    ).astype(np.uint8)
    image_l[mask_resized == 0] = 0

    glcm_l = skimage.feature.graycomatrix(
        image_l,
        distances=distances,
        angles=angles,
        symmetric=symmetric,
        normed=normed,
    )
    glcm_l_props = {
        p_name: skimage.feature.graycoprops(glcm_l, prop=p_name) for p_name in p_names
    }
    glcm_l_dict = build_texture_dict(glcm_l_props, distances, angles, 'L')

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
            **colors_rgb_dict,
            **colors_hsv_dict,
            **colors_lab_dict,
            **glcm_gray_dict,
            **glcm_l_dict,
        ),
        index=[image_name],
    )

    return measurements


def single_wrapped(args):
    return run_single(*args[0:-1]), args[-1]


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
            file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
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
@option_group(
    'Color options',
    option(
        '-n',
        '--n_colors',
        type=int,
        default=5,
        help='Number of dominant colors to ectract.',
        show_default=True,
    ),
)
@option_group(
    'Texture options',
    option(
        '-d',
        '--distances',
        multiple=True,
        default=(5, 7),
        help='Distances for gray level co-occurrence matrix calculation.',
        show_default=True,
    ),
    option(
        '-a',
        '--angles',
        multiple=True,
        default=(0, 45, 90),
        help='Angles for gray level co-occurrence matrix calculation.',
        show_default=True,
    ),
    option(
        '-w',
        '--resize_width',
        type=int,
        default=None,
        help='''
            Resize image and mask before calculating the gray level co-occurence matrix
            so that the width is -w/--resize_width while preserving aspect ratio.
        ''',
    ),
    option(
        '-h',
        '--resize_height',
        type=int,
        default=None,
        help='''
            Resize image and mask before calculating the gray level co-occurence matrix
            so that the height is -h/--resize_height while preserving aspect ratio.
        ''',
    ),
)
@option_group(
    'Output options',
    option('-g', '--group', type=str, default='', help='Group name for measurements.'),
)
@help_option('-h', '--help')
def single(
    image_file: Path,
    mask_file: Path,
    out_file: Optional[Path],
    n_colors: int,
    distances: List[int],
    angles: List[int],
    resize_width: Optional[int],
    resize_height: Optional[int],
    group: str,
):
    image_name = image_file.with_suffix('').name
    if out_file is None:
        out_file = Path(f'{image_name}_measurements.csv')

    measurements = run_single(
        image_file=image_file,
        mask_file=mask_file,
        n_colors=n_colors,
        distances=distances,
        angles=angles,
        resize_width=resize_width,
        resize_height=resize_height,
    )
    measurements['group'] = group
    measurements.insert(3, 'group', measurements.pop('group'))
    measurements.to_csv(out_file, index=False, mode='a', header=not out_file.exists())
    measurements.to_csv(out_file, index=False)


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
        help='''
            Input image directory or align output directory. If image_dir is an align
            output directory, -m/--mask_dir should be omitted.
        ''',
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
        help='''
            Input mask directory. Can be omitted if -i/--image_dir is an align output directory.
            Mask files must be PNG files named <IMAGE_NAME>_mask.png; i.e., for an image file
            "image_1.jpg" the corresponding mask must be named "image_1_mask.png".
        ''',
        required=False,
    ),
    option(
        '-o',
        '--out_file',
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f'''
            Output (CSV) file. By default a file with the name "<IMAGE_DIR>_measurements.csv"
            will be created in the current working directory ({Path(".").resolve()}).
        ''',
        default=None,
        required=False,
    ),
)
@option_group(
    'Color options',
    option(
        '-n',
        '--n_colors',
        type=int,
        default=5,
        help='Number of dominant colors to extract.',
        show_default=True,
    ),
)
@option_group(
    'Texture options',
    option(
        '-d',
        '--distances',
        multiple=True,
        default=(5, 7),
        help='Distances for gray level co-occurrence matrix calculation.',
        show_default=True,
    ),
    option(
        '-a',
        '--angles',
        multiple=True,
        default=(0, 45, 90),
        help='Angles for gray level co-occurrence matrix calculation.',
        show_default=True,
    ),
    option(
        '-w',
        '--resize_width',
        type=int,
        default=None,
        help='''
            Resize images and masks before calculating the gray level co-occurence matrix
            so that the width is -w/--resize_width while preserving aspect ratio. By default
            no resizing is applied.
        ''',
    ),
    option(
        '-h',
        '--resize_height',
        type=int,
        default=None,
        help='''
            Resize images and masks before calculating the gray level co-occurence matrix
            so that the height is -h/--resize_height while preserving aspect ratio. By default
            no resizing is applied.
        ''',
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
    show_default=True,
)
@help_option('-h', '--help')
def multi(
    image_dir: Path,
    mask_dir: Optional[Path],
    out_file: Optional[Path],
    n_colors: int,
    distances: List[int],
    angles: List[int],
    resize_width: Optional[int],
    resize_height: Optional[int],
    n_proc: int,
):
    import numpy as np
    from ..defaults import IMAGE_EXTENSIONS

    image_dir_name = image_dir.name
    if out_file is None:
        out_file = Path(f'{image_dir_name}_measurements.csv')

    # image_dir and mask_dir provided
    if not mask_dir is None:
        image_extensions = IMAGE_EXTENSIONS
        image_files = np.array(
            [f for f in image_dir.glob('*') if f.suffix.lower() in image_extensions]
        )
        image_names = np.array([f.with_suffix('').name for f in image_files])
        mask_files = np.array([mask_dir / f'{n}_mask.png' for n in image_names])
        mask_files_matching = np.array([f.exists() for f in mask_files])

        if mask_files_matching.sum() < 1:
            sys.stderr.write(f'Could not find any mask in mask_dir ({mask_dir}).\n')
            sys.exit(1)
        if len(mask_files_matching) - mask_files_matching.sum() > 0:
            msg = 'WARNING: Could not find mask files for images\n\t' + '\n\t'.join(
                sorted(image_names[~mask_files_matching])
            )
            print(msg)
        mask_files = mask_files[mask_files_matching]
        image_files = image_files[mask_files_matching]
        groups = np.array(['' for i in range(len(mask_files))])
    # only image_dir provided -> expecting a align output directory
    else:
        print('NOTE: only image_dir provided. Expecting an align output directory...')
        dirs = [
            f for f in image_dir.iterdir() if f.is_dir() and not f.name.startswith('.')
        ]
        image_files = np.array([])
        mask_files = np.array([])
        groups = np.array([])
        for d in dirs:
            img_dir = d / 'extractions'
            mask_dir = d / 'masks'

            cur_img_files = np.array(list(img_dir.glob('*.png')))
            cur_image_names = np.array([f.with_suffix('').name for f in cur_img_files])
            cur_mask_files = np.array(
                [mask_dir / f'{n}_mask.png' for n in cur_image_names]
            )
            cur_mask_files_matching = np.array([f.exists() for f in cur_mask_files])

            if cur_mask_files_matching.sum() < 1:
                sys.stderr.write(f'Could not find any mask in mask_dir ({mask_dir}).\n')
                sys.exit(1)
            if len(cur_mask_files_matching) - cur_mask_files_matching.sum() > 0:
                msg = 'WARNING: Could not find mask files for images\n\t' + '\n\t'.join(
                    sorted(cur_image_names[~cur_mask_files_matching])
                )
                print(msg)

            image_files = np.append(image_files, cur_img_files[cur_mask_files_matching])
            mask_files = np.append(mask_files, cur_mask_files[cur_mask_files_matching])
            groups = np.append(
                groups, [d.name for i in range(cur_mask_files_matching.sum())]
            )
    # prepare arguments for parallel processing
    symmetric = True
    normed = True
    args_list = [
        (
            image_file,
            mask_file,
            n_colors,
            symmetric,
            normed,
            distances,
            angles,
            resize_width,
            resize_height,
            group,  # need to pass groups since we are using imap_unordered for parallel processing
        )
        for image_file, mask_file, group in zip(image_files, mask_files, groups)
    ]

    # parallel runs
    import multiprocessing
    import pandas as pd
    from tqdm import tqdm

    with multiprocessing.Pool(processes=n_proc) as pool:
        if out_file.exists():
            out_file.unlink()

        for measurements, group in tqdm(
            pool.imap_unordered(single_wrapped, args_list),
            total=len(image_files),
        ):
            measurements['group'] = group
            measurements.insert(3, 'group', measurements.pop('group'))
            measurements.to_csv(
                out_file, index=False, mode='a', header=not out_file.exists()
            )


@group(
    'measure',
    help='Calculate several different measurements for image-mask pairs.',
    no_args_is_help=True,
)
@help_option('-h', '--help')
def measure():
    pass


def profile():
    run_single


measure.add_command(single)
measure.add_command(multi)
