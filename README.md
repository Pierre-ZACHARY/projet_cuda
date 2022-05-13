Rendu du projet CUDA

Membres du groupe : 
- Pierre ZACHARY 2183251
- Eoghann VEAUTE 2181406


Consignes générales :
- [x] Gestion des erreurs 
- [x] Utilisation des événements CUDA
- [x] Utilisation de std Chrono
- [x] Utilisation de différentes images
- [x] Analyse des différentes versions
- [x] Tester différentes taille de grille 2D 
- [ ] Le projet comporte un Makefile et a été testé sur les machines de la fac 
  - Notre projet comporte un CMakeFile, qui génère un makefile via la commande ccmake ( demande d'avoir Cmake d'installer sur la machine ) 
  - Nous n'avons pas pu tester le projet sur une machine de la fac ( voir le paragraphe ci-dessous ) 


Le projet a été testé sur un container docker avec la configuration suivante :
- Ubuntu 18
- Opencv ( lastest ) 
- Cuda 11.4
- Cmake ( lastest )
/!\ Le projet n'a pas ( pu ) être testé sur une machine de la fac, cependant elles sont censées avoir des configurations similaires à celle du container docker. Dans le cas où cela ne fonctionnerai pas avec une machine de la fac, vous pouvez toujours executez main sur docker en suivant le petit tutoriel ci-dessous.

Le projet a été testé avec le matériel suivant 
- rtx 3060 ti
- 940mx

Comment executer le Projet sur docker :

Docker installation : https://www.docker.com/get-started/ 

Etape 1 : Build le container docker : 
 - sur clion via la config "Dockerfile Build"
 - sinon via les commandes : 
   - docker build -t clion/remote-cuda-env:1.0 -f Dockerfile .
   - docker run -d --cap-add sys_ptrace -p127.0.0.1:2222:22 --name clion_remote_env clion/remote-cuda-env:1.0

Etape 2 : Run le projet dans le container :
- sur clion via la config "projet-cuda"
  - Si cette configuration n'apparait pas, soyez sur d'avoir la toolchain docker d'activé dans les settings de clion puis Build > Toolchains, soyez sur mettre l'image docker que vous venez de créer en Image de cette toolchain ( par défaut pour ce projet, l'image créée devrait s'appeler "projet_cuda" ), et il faut ajouter le paramètres **--gpus all** dans container settings
- sinon vous devez vous connectez en ssh, créer un build via le cmake file, push le build sur le container, puis make et run 
- ssh test@localost ( mdp : test ), vous pouvez push le contenu de projet_cuda dans /tmp sur docker, puis dans /tmp vous créer un dossier build et vous faite cmake ../. dedans, ça vous génère un makefile et tout normalement
