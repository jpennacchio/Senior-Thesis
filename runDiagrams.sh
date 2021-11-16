#!/bin/bash

cd 02_results

python3 ../01_src/makePhaseDiagrams.py 0 4 40 first_OMP.out second_OMP.out
python3 ../01_src/convert_text_to_image.py first_OMP.out second_OMP.out

python3 ../01_src/makePhaseDiagrams.py 1 4 40 first_LP.out second_LP.out
python3 ../01_src/convert_text_to_image.py first_LP.out second_LP.out

python3 ../01_src/makePhaseDiagrams.py 2 4 40 first_LASSOLAR.out second_LASSOLAR.out
python3 ../01_src/convert_text_to_image.py first_LASSOLAR.out second_LASSOLAR.out

python3 ../01_src/makePhaseDiagrams.py 3 4 30 first_LASSOQP.out second_LASSOQP.out
python3 ../01_src/convert_text_to_image.py first_LASSOQP.out second_LASSOQP.out

cd ..