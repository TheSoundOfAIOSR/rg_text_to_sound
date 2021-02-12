# Git Tutorial
## TL;DR
Here we give some guidelines on how to manage and contribute your code to the [Sound of AI Github](https://github.com/TheSoundOfAIOSR/thesoundofaiosr.github.io).
### Preliminaries
This document is in the making- we welcome suggestions for changes! Links currently refer to [my personal git repo](https://github.com/toedtli/soundofai), but tell me if this is still so when it shoudldn't anymore!

## Structure of the sound of ai git
In the text2sound working group, we'll have the following directory structure:

      └───text2sound-subgroup
            ├───playground
            │   ├───user1
            │   └───user2
            └───production
     └───other subgroups 
### Where to place your code
- Please place all your code in your own user directory. 
- You are free to manage your subfolder as you like, but please be careful not to push changes to other user's folders!
- Please create a Readme.md file in your directory- a text file describing the contents of your subfolder. 

### Getting started with git - experts skip this
Git is a widely used versioning control system. [Lots of tutorials](https://www.freecodecamp.org/news/learn-the-basics-of-git-in-under-10-minutes-da548267cc91/) exist. If you have any troubles starting (and are part of the sound of ai community), [contact me](beat.toedtli@ost.ch) any time.

### Setting up and Using the sound-of-ai repo: useful commands for everyone
Please use a [cookiecutter](https://cookiecutter.readthedocs.io/en/1.7.2/) template for your directory structure if possible. The [data science template](https://drivendata.github.io/cookiecutter-data-science/) might be useful:

	#activate your local virtual environment if you have one, then:
	cd text2sound-subgroup\playground\<firstname_lastname>
	pip install cookiecutter
	cookiecutter https://github.com/drivendata/cookiecutter-data-science

You'll next fill in some data. What they call `repo_name` will be the name of your new subfolder, `project_name` will be in the title of the README.md of your new subfolder. Please have a look at that file's content. If suitable, follow their suggestions. 

#### Git commands
Please configure your local git installation to indicate your name and email addess:

    git config --global user.name "<Your-Full-Name>"
    git config --global user.email "your@email.com"

Then clone the existing repository into your desired folder

    cd your_desired_folder
    git clone git@github.com:toedtli/soundofai.git
    
you'll get a subfolder `soundofai`. Add your directory (and files) at the right location:

    cd text2sound-subgroup\playground
    mkdir <yourfamilyname>

Commiting files is done by "adding" them locally to your staging area, committing them into your local repository, then pushing them to the remote repository:

    cd <yourfamilyname>
    # (create some files here, e.g. embeddings.ipynb)
    git add embeddings.ipynb
    
At this point it is useful use `git status` to check that you are *not committing files you should not change*, e.g. files in other user's subdirectories.

    git commit -m "Write your commit message here"
    git push

If you get an error message 

    fatal: The current branch main has no upstream branch.
    To push the current branch and set the remote as upstream, use

    git push --set-upstream origin main`
    git push

then this is your chance to give your branch a meaningful name! use `yourname_topic`, and enter 
    
    git push --set-upstream origin replacethiswithyourfamilyname_replacethiswithausefulstring

If you get

    git@github.com: Permission denied (publickey).
    fatal: Could not read from remote repository.

    Please make sure you have the correct access rights
    and the repository exists.

please [contact us](beat.toedtli@ost.ch). 

#### Working with branches
To make sure you're working on your entirely separate code base, with no danger of pushging file changes for files other authors are working on, branches are useful! To create your own branch of a repository:

    git checkout -b mycrazytest

Now you can check you're working on that new branch where you can do whatever you want:

    git branch

Committing works fine, but to push, you'll need to follow the advice given in the error message after `git push`:

    git push --set-upstream origin mycrazytest

Now when you're satisfied and your crazy idea is production ready, you might want to merge your branch `mycrazytest` into the main (or master) branch:

    git checkout main
    git merge mycrazytest

To delete the branch mycrazytest (make sure everything is merged into the main branch first), you might have to do it both locally and remotely. To remove the local branch:

     git branch -d mycrazytest

and to remove the remote branch:
    
     git push origin --delete mycrazytest

You may check that the branch is gone, say, on github.

