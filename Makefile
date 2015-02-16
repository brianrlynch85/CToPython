
DIR_BASE := $(CURDIR)

CC     := g++
CFLAGS := -g -Wall -std=c++11
SHM_IN_FLAG := -D _DEVICE_NAME_LOCK_PATH_=\".\" -D _SHM_BASEFILE_=\"$(CURDIR)/CAM\"
IDFLAGS = -I$(CURDIR)

all:
	$(CC) $(CFLAGS) -o single_parametric single_parametric.cpp vd_shm.c co_options.c $(SHM_IN_FLAG)
	@echo "Successfully compiled!"

shm_servers:
	@touch  $(DIR_BASE)/CAM

clean_shm:
	@rm -rf  $(DIR_BASE)/CAM
	@chmod u+x clean_shm.sh
	@./clean_shm.sh


