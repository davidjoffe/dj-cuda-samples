#!/bin/sh
# dj-cuda-samples — https://github.com/davidjoffe/dj-cuda-samples
# (c) David Joffe / DJ Software — BSL license, see LICENSE
#
# Helper script to try help auto-generate new projects from templates or other samples/projects as starting point
# (This script is not used or important for building or running samples.)
# WIP

DJTEMPLATE=$1
DJNEW=$2
DJNAME=$3
DJOLDNAME=$4
DJVERSION=$5

# shortname newproject "New Name" "Old Name" newprojectversion
# e.g. this is the idea roughly:
#./make_from_template.sh template_minimal template_gl "OpenGL Template" "DJ Sample" 0.1.0
#./make_from_template.sh template_gl template_gl_cuda "CUDA and OpenGL Template" "OpenGL Template" 0.1.0
#./make_from_template.sh template_gl_cuda molecular_sim  "Molecular Sim"   "CUDA and OpenGL Template" 0.1.0
#./make_from_template.sh template_gl galaxy_explorer     "Galaxy Explorer" "OpenGL Template" 0.1.0

echo DJTEMPLATE=$DJTEMPLATE
echo DJNEW=$DJNEW
echo DJNAME=$DJNAME
echo DJOLDNAME=$DJOLDNAME
echo DJVERSION=$DJVERSION

echo ===== dj-make-from-template $@
echo mkdir -p $DJNEW
mkdir -p $DJNEW
echo cp -rf $DJTEMPLATE/* $DJNEW
cp -rf $DJTEMPLATE/* $DJNEW

echo Replace name and other properties with $DJNEW $DJNAME
cat $DJTEMPLATE/README.md | sed -e "s/$DJOLDNAME/$DJNAME/g" -e "s/DJ Sample/$DJNAME/g" > $DJNEW/README.md

cat $DJTEMPLATE/CMakeLists.txt | sed -e "s/dj$DJTEMPLATE/dj$DJNEW/g" -e "s/VERSION [^ ]\+ /VERSION $DJVERSION /g" > $DJNEW/CMakeLists.txt

# std::cout << "djtemplate_minimal v1.0 running" << std::endl;
# std::cout << "djtemplate_minimal DJ Sample v1.0 running" << std::endl;
# become
# std::cout << "djgalaxy_explorer Galaxy Explorer v1.0 running" << std::endl;
cat $DJTEMPLATE/src/main.cpp | sed -e "s/dj$DJTEMPLATE $DJOLDNAME/dj$DJNEW $DJNAME/g" -e "s/dj$DJTEMPLATE /dj$DJNEW $DJNAME /g" -e "s/DJ Sample/$DJNAME/g" -e "s/v1\.0 /v$DJVERSION /g" > $DJNEW/src/main.cpp
