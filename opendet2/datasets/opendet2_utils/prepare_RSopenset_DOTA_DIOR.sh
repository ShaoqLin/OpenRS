DATA_DIR=datasets/dota_dior
DOTA_DIR=datasets/DOTA1024
DIOR_DIR=datasets/DIOR

# make neccesary dirs
rm $DATA_DIR -rf
echo "making dirs: /datsets/dota_dior"
mkdir -p $DATA_DIR
mkdir -p $DATA_DIR/Annotations
# mkdir -p DATA_DIR/JPEGImages
mkdir -p $DATA_DIR/ImageSets
mkdir -p $DATA_DIR/ImageSets/Main

# generate imageset
echo "generating DIOR sub imagesets"
# class incremental setting
# 10-15
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_specific.py --dir /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior --in_split DIOR_train --out_split DIOR_train_cls_spe_10_15 --start_class 10 --end_class 15 --pre_num_sample 8000 --post_num_sample 5000
# 10-20
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_specific.py --dir /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior --in_split DIOR_train --out_split DIOR_train_cls_spe_10_20 --start_class 10 --end_class 20 --pre_num_sample 8000 --post_num_sample 5000

# image incremental settings
# 500
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior --in_split DIOR_train --out_split DIOR_train_cls_agn_500 --start_class 10 --end_class 20  --post_num_sample 500
# 1000
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior --in_split DIOR_train --out_split DIOR_train_cls_agn_1000 --start_class 10 --end_class 20  --post_num_sample 1000
# 1500
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior --in_split DIOR_train --out_split DIOR_train_cls_agn_1500 --start_class 10 --end_class 20  --post_num_sample 1500
# all-1968
python datasets/opendet2_utils/prepare_openset_voc_coco_cls_agnostic.py --dir /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior --in_split DIOR_train --out_split DIOR_train_cls_agn_all --start_class 10 --end_class 20  --post_num_sample 20000

# copy DOTA data
echo 'copying DOTA images'
cp $DOTA_DIR/trainval/images $DATA_DIR/JPEGImages -r

# copy and convert DOTA annotation
echo "convert DOTA(coco) annotation to DOTA(voc)"
python datasets/opendet2_utils/convert_coco_to_voc.py --dir $DATA_DIR --ann_path $DOTA_DIR/trainval/DOTA_trainval1024.json

# copy DIOR images
echo 'copying DIOR images...'
find /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/DIOR/JPEGImages-trainval -name "*.jpg" -exec cp {} /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/JPEGImages \;
find /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/DIOR/JPEGImages-test -name "*.jpg" -exec cp {} /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/JPEGImages \;

# copy DOTA images
echo 'copying DOTA images...'
find /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/DOTA1024/trainval/images -name "*.png" -exec cp {} /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/JPEGImages \;
find /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/DOTA1024/test/images -name "*.png" -exec cp {} /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/JPEGImages \;
# convert png file to jpg file
# ls -1 *.png | xargs -n 1 bash -c 'convert "$0" "${0%.png}.jpg"'

# echo "copy voc imagesets"
# cp $VOC07_DIR/ImageSets/Main/train.txt $DATA_DIR/ImageSets/Main/voc07train.txt
# cp $VOC07_DIR/ImageSets/Main/val.txt $DATA_DIR/ImageSets/Main/voc07val.txt
# cp $VOC07_DIR/ImageSets/Main/test.txt $DATA_DIR/ImageSets/Main/voc07test.txt
# cp $VOC12_DIR/ImageSets/Main/trainval.txt $DATA_DIR/ImageSets/Main/voc12trainval.txt

echo "generate dota_dior_val imagesets"
cat /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/DIOR_val.txt > /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/dota_dior_val.txt
cat /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/DOTA_trainval1024.txt >> /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/dota_dior_val.txt

echo "genarate dota_dior_10_15_test imagesets"
cat /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/DIOR_train_cls_spe_10_15.txt > /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/dior_10_15_test.txt

echo "genarate dota_dior_10_15_test imagesets"
cat /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/DIOR_train_cls_spe_10_20.txt > /mnt/bee9bc2f-b897-4648-b8c4-909715332cb4/linshaoqing/data/datasets/dota_dior/ImageSets/Main/dior_10_20_test.txt


echo "generate voc_coco_20_40_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20_40_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_spe_20_40.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20_40_test.txt

echo "generate voc_coco_40_60_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20_60_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_spe_20_60.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20_60_test.txt

echo "generate voc_coco_60_80_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20_80_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_spe_20_80.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20_80_test.txt

echo "generate voc_coco_2500_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_2500_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_2500.txt >> $DATA_DIR/ImageSets/Main/voc_coco_2500_test.txt

echo "generate voc_coco_5000_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_5000_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_5000.txt >> $DATA_DIR/ImageSets/Main/voc_coco_5000_test.txt

echo "generate voc_coco_10000_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_10000_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_10000.txt >> $DATA_DIR/ImageSets/Main/voc_coco_10000_test.txt

echo "generate voc_coco_20000_test imagesets"
cat $DATA_DIR/ImageSets/Main/voc07test.txt > $DATA_DIR/ImageSets/Main/voc_coco_20000_test.txt
cat $DATA_DIR/ImageSets/Main/instances_train2017_cls_agn_20000.txt >> $DATA_DIR/ImageSets/Main/voc_coco_20000_test.txt

