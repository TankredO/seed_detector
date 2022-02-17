import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from cloup import command, option, help_option, Path as PathType, Choice, option_group


@command('single', help='Detect seeds in a single image.')
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
        '-o',
        '--out_dir',
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f'''
            Output directory. Will be created if it does not exist. By default
            a directory with the same name as IMAGE_FILE (but without suffix)
            will be created in the current working directory ({Path(".").resolve()}).
        ''',
        default=None,
        required=True,
    ),
)
@option(
    '-c',
    '--background_color',
    type=str,
    default=None,
    help='''
        Background color. Can be passed as comma-separated list of RGB values
        (e.g., "12, 52, 128") or alternatively as hex color code (e.g., "#04245f").
        By default the most common color range will be assumed to be the background
        color.
    ''',
)
# @option(
#     '-r',
#     '--refine',
#     type=bool,
#     is_flag=True,
#     help='''
#         Use CascadePSP for segmentation refinement.
#     ''',
# )
# @option(
#     '-d',
#     '--refinement_device',
#     type=str,
#     default='cuda:0',
#     show_default=True,
#     help='''
#         Refinement device for CascadePSP. Only relevant if -r/--refine is set.
#         Either "cuda:0", or "cuda:1", ..., to use a CUDA device, or alternatively
#         "cpu" to use the CPU. NOTE: Running CascadePSP using the CPU is very slow.
#     ''',
# )
# @option(
#     '-c',
#     '--refinement_cycles',
#     type=int,
#     default=1,
#     show_default=True,
#     help='''
#         Number of refinement cycles.
#     ''',
# )
@option_group(
    'Output options',
    option(
        '-m',
        '--masks_output',
        is_flag=True,
        help='Generate mask output for every detected object.',
    ),
    option(
        '-b',
        '--bbox_output',
        is_flag=True,
        help='Extract bounding boxes for every detected object.',
    ),
    option(
        '-p',
        '--padding',
        type=int,
        default=0,
        help='Padding around objects for mask and bounding box outputs.',
        show_default=True,
    ),
    option(
        '--rm_bg',
        is_flag=True,
        help='Set background pixel in bbox output to 0 (black).',
    ),
)
@help_option('-h', '--help')
def single(
    image_file: Path,
    out_dir: Optional[Path],
    background_color: str,
    # refine: bool,
    # refinement_device: str,
    # refinement_cycles: int,
    masks_output: bool,
    bbox_output: bool,
    padding: int,
    rm_bg: bool,
):
    image_name = image_file.with_suffix('').name

    # prepare output files
    if out_dir is None:
        out_dir = Path(image_name)
    out_file_mask = (out_dir / f'{image_name}_mask').with_suffix('.png')
    # out_file_mask_refined = (out_dir / f'{image_name}_mask_refined').with_suffix('.png')

    # parse background color string
    bg_col: Optional[Iterable[int]] = None
    if not background_color is None:
        background_color = ''.join(background_color.split())
        if ',' in background_color:
            try:
                bg_col = [int(s) for s in background_color.split(',')][:3]
            except:
                sys.stderr.write('ERROR: could not parse background_color.')
                sys.exit(1)
        elif '#' in background_color:
            try:
                bg_col = [b for b in bytes.fromhex(background_color[1:])]
            except:
                sys.stderr.write('ERROR: could not parse background_color.')
                sys.exit(1)
        else:
            sys.stderr.write('ERROR: invalid background_color option.')
            sys.exit(1)
        if any([x > 255 or x < 0 for x in bg_col]):
            sys.stderr.write('ERROR: color values must be in the range 0..255.')
            sys.exit(1)

    import skimage.io
    import cv2
    from ..tools import (
        segment_image_adaptive,
        get_minsize_adaptive,
        get_minsize_adaptive2,
        filter_bin_image,
        get_contours,
        resample_polygon,
        extract_subimage,
    )

    image = skimage.io.imread(image_file)

    bin_image = segment_image_adaptive(image, bg_col=bg_col, scale=None)
    min_size = get_minsize_adaptive2(bin_image)
    area_threshold = 512  # size of holes that will be filled within objects
    bin_image = filter_bin_image(
        bin_image, min_size=min_size, area_threshold=area_threshold
    )

    contours = get_contours(bin_image=bin_image, min_size=min_size)
    n_vertices = 500
    contours = [resample_polygon(c, n_vertices) for c in contours]

    out_dir.mkdir(parents=True, exist_ok=True)

    if masks_output:
        out_dir_masks = out_dir / 'masks'
        out_dir_masks.mkdir(parents=True, exist_ok=True)

    if bbox_output:
        out_dir_extractions = out_dir / 'extractions'
        out_dir_extractions.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_file_mask), bin_image * 255)

    if masks_output or bbox_output:
        for i, contour in enumerate(contours):
            sub_image, mask, bbox = extract_subimage(
                contour,
                image,
                remove_background=rm_bg,
                padding=(padding, padding, padding, padding),
            )

            if masks_output:
                cv2.imwrite(
                    str((out_dir_masks / f'{image_name}_mask{i}').with_suffix('.png')),
                    mask * 255,
                )
            if bbox_output:
                cv2.imwrite(
                    str(
                        (out_dir_extractions / f'{image_name}_bbox{i}').with_suffix(
                            '.png'
                        )
                    ),
                    sub_image[:, :, [2, 1, 0]],
                )

    # if not refine:
    #     return

    # import numpy as np
    # import warnings

    # with warnings.catch_warnings(record=True) as w:
    #     from ..segmentation_refinement import Refiner

    #     refiner = Refiner(device=refinement_device)

    #     mask = (bin_image * 255).astype(np.uint8)
    #     for _ in range(refinement_cycles):
    #         mask = refiner.refine(image=image, mask=mask)
    #     cv2.imwrite(str(out_file_mask_refined), mask)
