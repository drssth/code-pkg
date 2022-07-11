import sys
import os
import os.path


def clone_git_repo(path, repo_url):
    msg = ''
    if os.path.exists(path):
        msg = 'local repo copy exists'
    else:
        lib_path = os.path.dirname(path)
        current_path = os.getcwd()
        os.chdir(lib_path)
        msg = 'clone repo ...'
        os.system('git clone {} {}'.format(repo_url, os.path.basename(path)))
        os.chdir(current_path)
    
    if os.path.exists(path):
        msg += ' local repo copy exists'
        
    return msg


if __name__ == '__main__':
    pass

