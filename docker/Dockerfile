# Copyright (C) 2016-2022 by the multiphenics authors
#
# This file is part of multiphenics.
#
# multiphenics is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# multiphenics is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with multiphenics. If not, see <http://www.gnu.org/licenses/>.
#

FROM quay.io/fenicsproject/dev
MAINTAINER Francesco Ballarin <francesco.ballarin@unicatt.it>

USER root
RUN apt-get -qq update && \
    apt-get -qq remove python3-pytest && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    pip3 -q install --upgrade flake8 pytest pytest-flake8 pytest-xdist && \
    cat /dev/null > $FENICS_HOME/WELCOME

USER fenics
COPY --chown=fenics . /tmp/multiphenics

USER root
WORKDIR /tmp/multiphenics
RUN python3 setup.py -q install

USER fenics
WORKDIR $FENICS_HOME
RUN mkdir multiphenics && \
    ln -s $FENICS_PREFIX/lib/python3.6/dist-packages/multiphenics*egg/multiphenics multiphenics/source && \
    mv /tmp/multiphenics/tests multiphenics/ && \
    mv /tmp/multiphenics/tutorials multiphenics

USER root
RUN rm -rf /tmp/multiphenics
