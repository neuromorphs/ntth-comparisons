import os

repo = "https://gitlab.com/spinnaker2/py-spinnaker2.git"
dir = "lib"

cmd = "git clone {} {}".format(repo, dir)
os.system(cmd)