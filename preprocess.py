import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Orientationd, Spacingd

from pipeline.config import case_ids


def transform(path, mask=False):
    pipeline = Compose([
        LoadImaged('image'), EnsureChannelFirstd('image'),
        Orientationd('image', axcodes='RAS'),
        Spacingd('image', pixdim=(1., 1., 1.), mode='nearest' if mask else 'bilinear'),
        EnsureTyped('image', dtype=torch.float32, track_meta=True),
    ])
    tensor = pipeline({'image': str(path)})['image']
    array = np.asarray(tensor)[0]
    return ((array > 0).astype(np.uint8) if mask else np.clip(array, -150, 200)), dict(tensor.meta)


def bounding_box(mask, margin):
    positions = np.nonzero(mask)
    if not len(positions[0]):
        raise ValueError('Empty tumour mask')
    starts = [max(0, int(axis.min()) - margin) for axis in positions]
    stops = [min(mask.shape[i], int(axis.max()) + margin + 1) for i, axis in enumerate(positions)]
    return tuple(slice(a, b) for a, b in zip(starts, stops)), starts


def save_study_array(array, meta, path, start=None):
    # Preserve the study's saved array-axis convention for checkpoint compatibility.
    affine = np.asarray(meta['affine']).copy()
    if start is not None:
        affine[:3, 3] += affine[:3, :3] @ np.asarray(start[::-1])
    image = nib.Nifti1Image(array.transpose(2, 1, 0), affine)
    spacing = meta.get('spacing', meta.get('pixdim', [1, 1, 1, 1])[1:4])
    image.header.set_zooms(tuple(float(x) for x in spacing))
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, path)


def process_case(raw, output):
    output.mkdir(parents=True, exist_ok=True)
    image, meta = transform(raw / 'imaging.nii.gz')
    save_study_array(image, meta, output / 'full_image.nii.gz')
    annotations = sorted((raw / 'instances').glob('tumor_instance-1_annotation-*.nii.gz'))
    if not annotations:
        raise FileNotFoundError(f'No tumour annotations in {raw}')
    masks = []
    for annotation in annotations:
        mask, _ = transform(annotation, mask=True)
        if mask.shape != image.shape:
            raise ValueError(f'Image/mask grid mismatch: {annotation}')
        number = annotation.name.split('annotation-')[1].split('.')[0]
        base = output / 'annotations' / ('annotation-' + number)
        roi, start = bounding_box(mask, 0)
        save_study_array(image[roi], meta, base.with_name(base.name + '_image.nii.gz'), start)
        mask_path = base.with_suffix('.nii.gz')
        save_study_array(mask[roi], meta, mask_path, start)
        masks.append(mask_path)
        if len(annotations) == 1:
            roi, start = bounding_box(mask, 12)
            save_study_array(image[roi], meta, output / 'image.nii.gz', start)
            save_study_array(mask[roi], meta, output / 'mask.nii.gz', start)
    if len(masks) >= 2:
        reference = sitk.ReadImage(str(output / 'full_image.nii.gz'))
        aligned = [sitk.Resample(sitk.ReadImage(str(p)), reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8) for p in masks]
        consensus = sitk.Cast(sitk.STAPLE(aligned, 1.0) >= .5, sitk.sitkUInt8)
        consensus.CopyInformation(reference)
        sitk.WriteImage(consensus, str(output / 'consensus.nii.gz'))
        roi, start = bounding_box(sitk.GetArrayFromImage(consensus), 12)
        # These headers match the original context-crop writer; voxel arrays drive the models.
        origin = tuple(o + i * s for o, i, s in zip(reference.GetOrigin(), start[::-1], reference.GetSpacing()))
        for source, name in [(reference, 'image'), (consensus, 'mask')]:
            cropped = sitk.GetImageFromArray(sitk.GetArrayFromImage(source)[roi])
            cropped.SetSpacing(reference.GetSpacing())
            cropped.SetDirection(reference.GetDirection())
            cropped.SetOrigin(origin)
            sitk.WriteImage(cropped, str(output / (name + '.nii.gz')))
        paths = {'image': 'full_image.nii.gz', 'mask': 'consensus.nii.gz'}
    else:
        paths = {'image': str(masks[0].relative_to(output)).replace('.nii.gz', '_image.nii.gz'),
                 'mask': str(masks[0].relative_to(output))}
    (output / 'radiomics_paths.json').write_text(json.dumps(paths, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Prepare the study CT, annotation ROIs and STAPLE context crops.')
    parser.add_argument('--raw', type=Path, default=Path('work/raw'))
    parser.add_argument('--output', type=Path, default=Path('work/preprocessed'))
    parser.add_argument('--include-feature-selection', action='store_true')
    parser.add_argument('--case-id')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    ids = case_ids(args.include_feature_selection)
    if args.case_id:
        if args.case_id not in ids:
            parser.error('--case-id is outside the selected cohort')
        ids = [args.case_id]
    for index, case in enumerate(ids, 1):
        destination = args.output / case
        if (destination / 'radiomics_paths.json').is_file() and not args.overwrite:
            continue
        print(f'[{index}/{len(ids)}] {case}', flush=True)
        process_case(args.raw / case, destination)


if __name__ == '__main__':
    main()
