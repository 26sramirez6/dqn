export DEBUG_BUILD="Debug"
export RELEASE_BUILD="RelWithDebInfo"
export FULL_RELEASE_BUILD="Release"

function usage(){
    echo "generate_build.sh --release or --debug"
    echo ""
    echo "./generate_build.sh"
    echo "-h  --help"
    echo "-r  --release with debug info <-- this is the previous default"
    echo "-f  --full (this is acutally release with no debugging)"
    echo "-d  --debug"
    echo ""
}

function go() {
  set -x  
  if [ ! -d libtorch ]
  then
	echo "libtorch not found, installing"
	wget https://download.pytorch.org/libtorch/cu102/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu102.zip
	unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cu102.zip
  fi
  if [ ! -d build ]
  then
    #rm -fr build
	mkdir -p build
  fi
  #mkdir -p build
  cd build
  #export CC=/usr/local/bin/gcc
  #export CXX=/usr/local/bin/g++
  #export GTEST_DIR=$PWD/../googletest
  #echo $GTEST_DIR
  export FLAGS="-lrt -fPIE"
  if [[ $1 == ${DEBUG_BUILD} ]]
  then
      export FLAGS="${FLAGS} -fno-inline-functions"
  else
      export FLAGS="${FLAGS} -DUSE_CUDA"
  fi
  echo $FLAGS

  cmake -DCMAKE_CXX_FLAGS="${FLAGS}" -DCMAKE_PREFIX_PATH="$PWD;$PWD/../libtorch;$PWD/../Eigen;$PWD/../googletest;" -DCMAKE_BUILD_TYPE=$1 -G "Unix Makefiles" ../
}

if [ "$1" == "" ]
then
  usage
else
  while [ "$1" != "" ]; do
      PARAM=`echo $1 | awk -F= '{print $1}'`
      VALUE=`echo $1 | awk -F= '{print $2}'`
      case $PARAM in
          -h | --help)
              usage
              exit
              ;;
          -d | --debug)
              go ${DEBUG_BUILD}
              ;;
          -r | --release)
              go ${RELEASE_BUILD}
              ;;
          -f | --full)
              go ${FULL_RELEASE_BUILD}
              ;;
          *)
              echo "ERROR: unknown parameter \"$PARAM\""
              usage
              exit 1
              ;;
      esac
      shift
  done
fi 

