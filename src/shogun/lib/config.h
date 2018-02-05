#ifndef __SHOGUN_LIB_CONFIG_H__
#define __SHOGUN_LIB_CONFIG_H__

#define MACHINE "x86_64"

#define SFMT_MEXP 19937
#define DSFMT_MEXP 19937

/* #undef HAVE_HDF5 */
/* #undef HAVE_CURL */
/* #undef HAVE_JSON */
/* #undef HAVE_XML */
/* #undef HAVE_DOXYGEN */
/* #undef HAVE_LAPACK */
/* #undef HAVE_MVEC */
/* #undef HAVE_PROTOBUF */
/* #undef HAVE_TFLOGGER */

/* #undef HAVE_ARPACK */
/* #undef HAVE_VIENNACL */
/* #undef VIENNACL_VERSION */
/* #undef HAVE_OPENCV */
/* #undef HAVE_CATLAS */
/* #undef HAVE_ATLAS */
/* #undef HAVE_MKL */
/* #undef HAVE_NLOPT */
/* #undef USE_LPSOLVE */
#define HAVE_PTHREAD 1
#define HAVE_OPENMP 1
/* #undef USE_CPLEX */
/* #undef HAVE_COLPACK */
/* #undef HAVE_ARPREC */
/* #undef USE_META_INTEGRATION_TESTS */

#define HAVE_LGAMMAL 1
/* #undef USE_LOGCACHE */
/* #undef USE_LOGSUMARRAY */

/* Tells ViennaCL to use OpenCL as computation backend */
#define VIENNACL_WITH_OPENCL 1

/* Eigen Lapack optimization flags */
/* #undef EIGEN_USE_BLAS */
/* #undef EIGEN_USE_LAPACKE */
/* #undef EIGEN_USE_LAPACKE_STRICT */
/* #undef EIGEN_USE_MKL_VML */
/* #undef EIGEN_USE_MKL_ALL */

/* for linear algebra global backend setups */
/* #undef USE_EIGEN3_GLOBAL */
/* #undef USE_VIENNACL_GLOBAL */

/* #undef USE_EIGEN3_LINSLV */
/* #undef USE_VIENNACL_LINSLV */

/* #undef USE_EIGEN3_EIGSLV */
/* #undef USE_VIENNACL_EIGSLV */

#define HAVE_DECL_SIGNGAM 1

#define HAVE_FDOPEN 1

#define USE_SHORTREAL_KERNELCACHE 1
#define USE_BIGSTATES 1

/* #undef USE_HMMDEBUG */
#define USE_HMMCACHE 1
/* #undef USE_HMMPARALLEL */
/* #undef USE_HMMPARALLEL_STRUCTURES */

/* #undef USE_PATHDEBUG */

#define USE_SVMLIGHT 1
/* #undef USE_MOSEK */
#define USE_GPL_SHOGUN 1

/* #undef USE_GLPK */
/* #undef USE_LZO */
/* #undef USE_GZIP */
/* #undef USE_BZIP2 */
/* #undef USE_LZMA */
/* #undef USE_SNAPPY */

#define HAVE_SSE2 1
#define HAVE_BUILTIN_VECTOR 1

/* #undef DARWIN */
/* #undef FREEBSD */
#define LINUX 1

/* #undef USE_SWIG_DIRECTORS */
/* #undef TRACE_MEMORY_ALLOCS */
/* #undef USE_JEMALLOC */

/* #undef HAVE_CXX0X */
#define HAVE_CXX11 1

/* does the compiler support abi::__cxa_demangle */
#define HAVE_CXA_DEMANGLE 1

/* random related defines */
/* #undef HAVE_ARC4RANDOM */
#define DEV_RANDOM "/dev/urandom"

#endif /* __SHOGUN_LIB_CONFIG_H__ */
