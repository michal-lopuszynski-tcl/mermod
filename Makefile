MODULE_NAME=mermod

PY_DIRS=src/mermod tests setup.py

PY_MYPY_FLAKE8=src/mermod tests setup.py

FILES_TO_CLEAN=src/mermod.egg-info dist

include Makefile.inc
