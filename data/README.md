# Data upload to MongoDB
Scripts to upload different data sources to MongoDB with a unified data spec. All examples below expect a local running MongoDB.
In case you want to upload to a different MongoDB or adjust the database and collection, use the parameters. Check out the `--help` for each script.

## label_spec.py
Data specification for semseg, depth maps, sensor specifics and object detection. All data from the different sources is transformed and converted to
fit this spec. These enables to train the same model from different data sources without having to write specific pre-processors for
each data source.

## Comma10k - Semseg
Semantic segmentation data from comma.ai (https://github.com/commaai/comma10k). Clone the repository to your machine and run
`>> comma10k.py --src_path /path/to/comma10k_repo`.

## Mapillary - Semseg
Semantic segmentation data from mappilary (https://www.mapillary.com/dataset/vistas). Download and extract then run:
`>> mapillary.py --src_path /path/to/mappilary_data --dataset training`

## Kitti - 2D and 3D Object Detection
Kitti contains about 7500 training images including 2D boxes and 3D data. Download from http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d.
You will need:
- Download left color images of object data set (12 GB) (data_object_image_2.zip)
- Download training labels of object data set (5 MB) (data_object_label_2.zip)
- Download camera calibration matrices of object data set (16 MB) (data_object_calib.zip)

```bash
# Run script to convert Kitti data to od_spec and upload to MongoDB
kitti.py --image_path /path/to/data_object_image_2/training/image_2 
         --label_path /path/to/data_object_label_2/training/label_2
         --calib_path /path/to/data_object_calib/training/calib
```

## NuImages - 2D Object Detection
Selected keyframes with 2D boxes and camera intrinsics. But no 3D info, real or cuboid like.
From https://www.nuscenes.org/download (You will need to register and login first) download:
- nuImages - All - Metadata
- nuImages - All - Samples
The samples have to be extracted within the Metadata folder. Folder structure should look something like this:
```
nuimages-v1.0-all
├── samples
|    ├── CAM_BACK
|    ├── CAM_BACK_LEFT
|    ├── ...
├── v1.0-train
|    ├── sample.json
|    ├── ...
├── v1.0-val
|    ├── sample.json
|    ├── ...
├── v1.0-test
|    ├── sample.json
|    ├── ...
└── v1.0-mini
     ├── sample.json
     ├── ...
```
```bash
# Run script to convert NuImage data to od_spec and upload to MongoDB
nuimage.py --path /path/to/nuimages-v1.0-all 
           --version v1.0-train
```

## NuScenes - 2D and 3D Object Tracking
Download: https://www.nuscenes.org/download (You will need to register and login first)
- Full dataset (v1.0) - Metadata
- Full dataset (v1.0) - File blobs part [1-10] - Keyframe blobs
Afterwards you will have to move everything into a folder named "samples", same as with the nuImages folder structure.
```bash
# Run script to convert NuScenes data to od_spec and upload to MongoDB
nuscenes.py --path /path/to/nuscenes-v1.0-all
```

# DrivingStereo - Depthmap
Download: https://drivingstereo-dataset.github.io/
- Training data - Left Images
- Training data - Depth Maps
Folder structure should have all scene folders (e.g. 2018-10-19-09-30-39) of depth maps in one folder and the same scene folders for the left images in another folder.
```bash
driving_stereo.py --depth_maps /path/to/depth_maps --images /path/to/images
```
