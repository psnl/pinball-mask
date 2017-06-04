# pinball-mask
Create a mask layer with opencv to be used with an automated pinball machine

## get
git clone https://github.com/psnl/pinball-mask.git

## build
* cd pinball-mask
* mkdir build
* cd build
* cmake ..
* make

## run
###### Default mask
./pinball-mask
###### Custom mask
./pinball-mask -i=<mask.png>

## markers
Print the pinball-mask.png in the markers directory on a sheet of paper to test the program.
