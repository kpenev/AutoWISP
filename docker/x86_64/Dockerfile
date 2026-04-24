FROM python:3.11-slim
RUN apt-get -y update && \
    apt-get -y --no-install-recommends install astrometry.net
RUN pip3 install autowisp
RUN mkdir -p /storage/autowisp_app_data

ENV XDG_DATA_HOME=/storage/autowisp_app_data
ENV WIN_PD_OVERRIDE_APPDATA=/storage/autowisp_app_data

EXPOSE 8089

CMD ["wisp-bui", "--hostname", "0.0.0.0:8089"]
