echo 'Clustering genes unsing MMSEQ2'
echo 'Input Folder: ' $1
echo 'Output Folder: ' $2

mkdir $2

echo ''
echo 'Step 1'
now=$(date);echo "$now";
python AddedSteps/FormatData.py $1 $2

echo ''
echo 'Step 2'
now=$(date);echo "$now";
mmseqs easy-linclust --threads 1 $2/PROTEIN.fa $2/OUT $2/tmp

echo ''
echo 'Step 3'
now=$(date);echo "$now";
python AddedSteps/Cluster2Table.py $1 $2

echo 'Done!'
now=$(date);echo "$now";
