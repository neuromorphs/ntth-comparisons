import os, shutil

def get_repo(repo, dir):
    get_repo = True

    if os.path.isdir(dir):
        check = input("Directory {} already exists, overwrite? [y/n] ".format(dir))
        if check[:1] == 'y':
            get_repo = True
            shutil.rmtree(dir)
        else:
            get_repo = False
    if get_repo:
        cmd = "git clone {} {}".format(repo, dir)
        os.system(cmd)
    

repo = "https://gitlab.com/spinnaker2/py-spinnaker2.git"
dir = "lib/py-spinnaker2"
cwd = os.getcwd()

get_repo(repo, dir)

setupfile = '{}/setup.py'.format(dir)

if os.path.isfile(setupfile):
    os.system('cd {}; pip install -e .; cd {}'.format(dir, cwd))
    
repo_s2 = "git@gitlab.com:spinnaker2/s2-sim2lab-app.git"
dir_s2 = "lib/s2-sim2lab-app"

get_repo(repo_s2, dir_s2)

for d in [dir_s2, os.path.join(dir_s2, 'chip/app-pe/s2-lib')]:
    os.system('cd {}; make; cd {}'.format(d, cwd))

os.system('cd {}; cmake CMakeLists.txt; make; cd {}'.format(os.path.join(dir_s2, 'host/experiment/app'), cwd))
os.system('cd {}; ./make_all_apps.py; cd {}'.format(os.path.join(dir_s2, 'host/experiment/'), cwd))


