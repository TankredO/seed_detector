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
    out_dir: Optional[Path],
    padding: int,
):
    image_name = image_file.with_suffix('').name
    if out_dir is None:
        out_dir = Path(f'{image_name}_aligned')

    import numpy as np
    import cv2
    import skimage.transform
    import pyefd
    from ..defaults import DEFAULT_N_POLYGON_VERTICES
    from ..tools import (
        get_contours,
        resample_polygon,
        rotate_upright,
        align_shapes,
        extract_subimage,
    )

    # read in image and mask
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(image_file))

    # get contours from mask
    n_vertices = DEFAULT_N_POLYGON_VERTICES
    min_size = 1  # we assume only perfect masks
    contours = get_contours(mask, min_size=min_size)
    contours = [
        resample_polygon(c, n_points=n_vertices).astype(np.float32) for c in contours
    ]

    # align contours/polygons
    # the first contour is used as reference
    # TODO: let user select a contour as reference?
    ref, _ = rotate_upright(contours[0])
    ref_approx = pyefd.reconstruct_contour(
        pyefd.elliptic_fourier_descriptors(ref, order=3, normalize=False),
        num_points=100,
    )

    # Rotate all other contours to best match the reference contour
    # Shapes are approximated using elliptic Fourier analysis
    contours_rot = []
    rots = []
    for contour in contours:
        contour_approx = pyefd.reconstruct_contour(
            pyefd.elliptic_fourier_descriptors(contour, order=3, normalize=False),
            num_points=100,
        )
        _, best_i, rotation, scale, _ = align_shapes(ref_approx, contour_approx)
        contour_approx_rot = contour_approx.dot(rotation.T)
        disparity = np.sum(np.square(ref_approx - contour_approx_rot))

        rots.append(np.arctan2(rotation.T[0, 0], rotation.T[1, 0]) * 180 / np.pi)
        contour_rot = contour.dot(rotation.T)
        contour_rot = np.r_[
            contour_rot[
                best_i:,
            ],
            contour_rot[
                :best_i,
            ],
        ]
        contours_rot.append(contour_rot)

    # extract objects and rotate them
    out_dir_extractions = out_dir / 'extractions'
    out_dir_extractions.mkdir(parents=True, exist_ok=True)
    out_dir_masks = out_dir / 'masks'
    out_dir_masks.mkdir(parents=True, exist_ok=True)

    for i, (contour, angle) in enumerate(zip(contours, rots)):
        sub_image, sub_mask, *_ = extract_subimage(contour, image)
        sub_mask_rot = skimage.transform.rotate(sub_mask, angle, resize=True, order=0)
        sub_image_rot = skimage.transform.rotate(sub_image, angle, resize=True, order=1)

        contour = resample_polygon(get_contours(sub_mask_rot, 1)[0], n_vertices)
        sub_image_rot, sub_mask_rot, *_ = extract_subimage(
            contour,
            sub_image_rot,
            padding=(padding, padding, padding, padding),
            remove_background=True,
        )

        import matplotlib.pyplot as plt

        out_file_image = out_dir_extractions / f'{image_name}_bbox{i}.png'
        cv2.imwrite(str(out_file_image), sub_image_rot * 255)

        out_file_mask = out_dir_masks / f'{image_name}_mask{i}.png'
        cv2.imwrite(str(out_file_mask), sub_mask_rot * 255)


def single_wrapped(args):
    run_single(*args)


@command(
    'single', help='Align contours and extract rotated subimages for a single image.'
)
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
        '--out_dir',
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f'''
            Output directory. Will be created if it does not exist. By default
            a directory with the name as "<IMAGE_FILE>_aligned" (but without suffix)
            will be created in the current working directory ({Path(".").resolve()}).
        ''',
        default=None,
        required=False,
    ),
)
@option_group(
    'Output options',
    option(
        '-p',
        '--padding',
        type=int,
        default=5,
        help='Padding around contours.',
        show_default=True,
    ),
)
@help_option('-h', '--help')
def single(
    image_file: Path,
    mask_file: Path,
    out_dir: Optional[Path],
    padding: int,
):
    run_single(
        image_file=image_file,
        mask_file=mask_file,
        out_dir=out_dir,
        padding=padding,
    )


@command(
    'multi', help='Align contours and extract rotated subimages for multiple images.'
)
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
        help='''
            Input mask directory or detection output directory. Mask files must
            be PNG files named <IMAGE_NAME>_mask.png; i.e., for an image file
            "image_1.jpg" the corresponding mask must be named "image_1_mask.png".
        ''',
        required=True,
    ),
    option(
        '-o',
        '--out_dir',
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f'''
            Output directory. Will be created if it does not exist. By default
            a directory with the name "<IMAGE_DIR>_aligned" will be created in
            the current working directory ({Path(".").resolve()}).
        ''',
        default=None,
        required=False,
    ),
)
@option_group(
    'Output options',
    option(
        '-p',
        '--padding',
        type=int,
        default=5,
        help='Padding around contours.',
        show_default=True,
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
    image_dir: Path,
    mask_dir: Path,
    out_dir: Optional[Path],
    padding: int,
    n_proc: int,
):
    import numpy as np
    from ..defaults import IMAGE_EXTENSIONS

    image_dir_name = image_dir.name
    if out_dir is None:
        out_dir = Path(f'{image_dir_name}_aligned')

    image_extensions = IMAGE_EXTENSIONS
    image_files = np.array(
        [f for f in image_dir.glob('*') if f.suffix.lower() in image_extensions]
    )
    image_names = np.array([f.with_suffix('').name for f in image_files])

    # check if mask_dir is detection dir
    dir_content = [f.name for f in mask_dir.iterdir()]
    dirs_matching = np.zeros((len(image_names)), dtype=bool)
    for i, image_name in enumerate(image_names):
        if image_name in dir_content:
            dirs_matching[i] = True
    # mask_dir is detection output
    if dirs_matching.sum() > 0:
        msg = (
            f'NOTE: Found {dirs_matching.sum()} directories in mask_dir ({mask_dir}) matching '
            f'image file names in image_dir ({image_dir}): '
            'assuming mask_dir to be a detection output directory.'
        )
        print(msg)
        if len(dirs_matching) != dirs_matching.sum():
            msg = (
                'WARNING: Could not find detection output for images\n\t'
                + '\n\t'.join(sorted(image_names[~dirs_matching]))
            )
            print(msg)

        mask_files = np.array(
            [
                mask_dir / image_name / f'{image_name}_mask.png'
                for image_name in image_names[dirs_matching]
            ]
        )
        for mask_file in mask_files:
            if not mask_file.exists():
                sys.stderr.write(f'ERROR: mask file {mask_file} does not exist.\n')
                sys.exit(1)
        image_files = image_files[dirs_matching]

    # mask_dir is a simple directory containing mask images.
    else:
        mask_files = np.array(
            [mask_dir / f'{image_name}_mask.png' for image_name in image_names]
        )
        mask_files_matching = np.zeros(len(image_names), dtype=bool)
        for i, mask_file in enumerate(mask_files):
            if mask_file.exists():
                mask_files_matching[i] = True
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

    # prepare arguments for parallel processing
    args_list = [
        (
            image_file,
            mask_file,
            out_dir.joinpath(image_file.with_suffix('').name),
            padding,
        )
        for image_file, mask_file in zip(image_files, mask_files)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    # parallel runs
    import multiprocessing
    from tqdm import tqdm

    with multiprocessing.Pool(processes=n_proc) as pool:
        list(
            tqdm(
                pool.imap_unordered(single_wrapped, args_list),
                total=len(image_files),
            )
        )


@group(
    'align', help='Align contours and extract rotated subimages.', no_args_is_help=True
)
@help_option('-h', '--help')
def align():
    pass


align.add_command(single)
align.add_command(multi)
