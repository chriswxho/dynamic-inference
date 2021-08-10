FROM ubuntu
# ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y git
RUN mkdir ~/.ssh
RUN echo 'Host github.com\n\tHostName github.com\n\tUser git\n\tIdentityFile /secrets/cho-ssh-key\n\tIdentitiesOnly yes' > ~/.ssh/config
RUN chmod 600 ~/.ssh/config
RUN ssh-keyscan github.com >> ~/.ssh/known_hosts