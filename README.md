# hal3d

hal3d (pronounced haled) is a three-dimensional Arbitrary Lagrangian-Eulerian code that solves Euler's inviscid compressible equations of hydrodynamics, including subcell swept edge remapping to support hourglass elimination

There is currently an effort to port the code to modern architectures and implement multi-material interfaces.

# Build

Before building the dependent `hal3d` application, it is necessary to clone the application into the `arch` project. The instructions can be found on the `arch` project README.

```
git clone git@github.com:uob-hpc/arch
cd arch
git clone git@github.com:uob-hpc/hal3d
cd hal3d
```

It should now be possible to change directory to hal3d and build.
