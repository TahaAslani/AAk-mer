echo 'Finding SNPs using Snippy'
echo 'Input Folder: ' $1
echo 'Output Folder: ' $2
echo 'Reference: ' $3
echo 'Number of CPUs: ' $4

mkdir $2

echo ''
echo 'Step 1'
now=$(date);echo "$now";
#python RunSnippy.py $1 $2 $3 $4 1 0

echo ''
echo 'Step 2'
now=$(date);echo "$now";
python CombineSNPs.py $1 $2

echo 'Done!'
now=$(date);echo "$now";
