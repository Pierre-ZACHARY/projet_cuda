Docker installation : https://www.docker.com/get-started/ 

Etape 1 : Build le container docker : 
 - sur clion via la config "Dockerfile Build"
 - sinon via les commandes : 
   - docker build -t clion/remote-cuda-env:1.0 -f Dockerfile .
   - docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 --name clion_remote_env clion/remote-cuda-env:1.0

Etape 2 : Run le projet dans le container :
- sur clion via la config "projet-cuda"
- sinon vous devez vous connectez en ssh, créer un build via le cmake file, push le build sur le container, puis make et run 