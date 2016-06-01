#!/bin/bash

rootdir=$(pwd)
tmpdir=vendortmp
rm -rf tiny_cnn $tmpdir
mkdir $tmpdir
cd $tmpdir

wget github.com/nyanp/tiny-cnn/archive/master.tar.gz
tar -xzf master.tar.gz
cp -R tiny-cnn-master/tiny_cnn $rootdir/

cd $rootdir
rm -rf $tmpdir

