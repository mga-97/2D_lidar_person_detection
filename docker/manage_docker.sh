#!/bin/bash

############################################################
# Help                                                     #
############################################################
usage()
{
    # Display Help
    echo "Syntax: ./build-docker.sh [options]"
    echo "options:"
    echo "    -b, --build                        Use the passed options to build a new image. If not passed, the options passed will be used to identify the image to run"
    echo "    -c, --cuda                         Build/run the image from the nvidia cuda official image ($CUDA_DEF) and with the '$CUDA_SUFFIX' tag"
    echo "    -r, --ros_distro + \"distro_name\"   Build/run the image with the passed distro (the passed value will be also used to compose the image tag)"
    echo "    -h, --help                         See current help"
    echo "If the parent image is not specified (neither -u nor -c), the '$UBUNTU_DEF' one will be used"
    echo "If the ROS2 distro is not specified, 'humble' will be used."
    echo "WARNING: If a wrong ROS2 distro name is passed, the image build will fail"

    exit
}

get_opts()
{
    while [[ $# -gt 0 ]]
    do
        key="$1"
        case $key in
            -b|--build)
                BUILD_IMAGE=true
                shift
                ;;
            -c|--cuda)
                IMAGE=$CUDA_DEF
                PARENT_SUFFIX=$CUDA_SUFFIX
                RUN_WITH_GPU=true
                shift
                ;;
            -r|--ros_distro)
                ROS_DISTRO=$1
                shift
                ;;
            -h|--help)
                usage
                exit
                ;;
        esac
    done
}

source ./docker_vars.sh
IMAGE=$UBUNTU_DEF
ROS_DISTRO=iron
PARENT_SUFFIX=$UBUNTU_SUFFIX
BUILD_IMAGE=false
RUN_WITH_GPU=false
get_opts $@

if [[ $BUILD_IMAGE == "true" ]]; then
    sudo docker build --progress plain --build-arg base_img=$IMAGE --build-arg ros_distro=$ROS_DISTRO -t person_det/r1_crowd:$PARENT_SUFFIX"_"$ROS_DISTRO .
else
    sudo xhost +
    if [[ $RUN_WITH_GPU == "true" ]]; then
        sudo docker run --rm -it --privileged --network host --pid host -e NVIDIA_DRIVER_CAPABILITIES=all -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -e QT_X11_NO_MITSHM=1 --gpus all person_det/r1_crowd:$PARENT_SUFFIX"_"$ROS_DISTRO
    else
        sudo docker run --rm -it --privileged --network host --pid host -e DISPLAY -e -v /tmp/.X11-unix:/tmp/.X11-unix -e QT_X11_NO_MITSHM=1 person_det/r1_crowd:$PARENT_SUFFIX"_"$ROS_DISTRO
    fi
fi
