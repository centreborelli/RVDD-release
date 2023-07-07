file1="../mInverseCompositional_1.00/data/homography1.png"
file2="../mInverseCompositional_1.00/data/homography2.png"

./tvl1flow $file1 $file2
/extra/Sources/imscript/bin/backflow flow.flo $file2 flow.png
vpv $file1 $file2 flow.png
