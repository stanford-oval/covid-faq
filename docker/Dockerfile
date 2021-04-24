# override this to "nvidia/cuda:10.1-runtime-ubi8" if cuda is desired
ARG BASE_IMAGE=registry.access.redhat.com/ubi8/ubi:latest
FROM ${BASE_IMAGE}

MAINTAINER Thingpedia Admins <thingpedia-admins@lists.stanford.edu>

RUN dnf -y module enable python38 \
        && dnf -y install python38 \
        python38-devel \
        python38-pip \
        python38-wheel \
        && dnf clean all

COPY . /opt/covid-faq
RUN pip3 install -e /opt/covid-faq \
        && rm -fr /root/.cache

RUN useradd -ms /bin/bash covid-faq
USER covid-faq

WORKDIR /home/covid-faq
ENTRYPOINT ["covidfaq"]