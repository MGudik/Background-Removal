FROM ambolt/emily:1.0.0-tf-cv

ARG DEBIAN_FRONTEND=noninteractive

# Install opencv and its dependencies
RUN echo "**** GIT ****" && \
  apt-get update; \
  apt-get install -y --no-install-recommends \
  git

RUN git clone https://github.com/Kamlesh364/Mask-RCNN-TF2.7.0-keras2.8.0.git Mask
COPY model.py Mask/mrcnn/model.py
RUN cd Mask && python setup.py install
RUN cd ..
RUN rm -rf Mask


WORKDIR /workspace

# Static project file that allows us to COPY, RUN and cache setup scripts conditionally
ARG EMILY_IDREQ=.emily/.pid

# Build arguments
# Provided by Emily when opening projects in PyCharm
ARG CONFIGURE_PYCHARM_SSH_SCRIPT

# Run and cache the SSH configuration required for Pycharm if the script is present
COPY $EMILY_IDREQ $CONFIGURE_PYCHARM_SSH_SCRIPT ./
RUN if [ ! -z "$CONFIGURE_PYCHARM_SSH_SCRIPT" ]; then \
  SCRIPT=$(basename "$CONFIGURE_PYCHARM_SSH_SCRIPT"); \
  PID=$(basename "$EMILY_IDREQ"); \
  sudo /bin/bash "$SCRIPT"; \
  sudo rm $SCRIPT; \
  sudo rm $PID; \
  fi;

# Add your changes here e.g. apt package installations
# RUN apt-get --allow-releaseinfo-change -y update && apt-get install -y #     package1 #     package2 ... 

# Create and set the project user
RUN adduser --disabled-password --gecos "" --home /home/Background-Remo --force-badname Background-Remo
RUN adduser Background-Remo sudo
RUN chown Background-Remo /workspace
RUN chown Background-Remo /home/Background-Remo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER Background-Remo

ENV PATH="/home/Background-Remo/.local/bin:$PATH"

# Install Pip requirements
COPY requirements.txt requirements.txt
RUN pip install --disable-pip-version-check -r requirements.txt

# Copy all directory contents (sans .dockerignore) into container
COPY . .

CMD ["/bin/bash", "run.sh"]
