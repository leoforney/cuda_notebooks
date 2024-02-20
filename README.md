# cuda-notebook

An expiremental notebook to log my progress as I learn cuda.

To run, build with cmake by running

`mkdir build && cd build`

then 

`cmake ..`

After that use your preferred build system configured through cmake. Ex: `make`

Then run the main starter program by using `./cuda-notebooks <program name>`

Currently, there are the following programs made:

## vector-operations - `vo`
Simple vector operations by creating random vector float arrays and doing addition/subtraction/dot product

Flags:
- `--numElements` or `--n` specify an amount of elements per vector to generate

## image-processing - `ip`
Image processing, currently converts image to grayscale

Flags:
- `--file` or `--f` specify a different input file other than image.jpg
- `--output` or `--of` specify a different output file location other than image_grayscale.jpg