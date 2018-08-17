Project I: Finding Lane Lines on the Road (F.L.L.R.)
======================================================

### 1. Building a portfolio

[![find lane lines](http://img.youtube.com/vi/q8ZysjEREiM/0.jpg)](https://youtu.be/q8ZysjEREiM "find lane lines")

### 2. Project introduction

**Finding Lane Lines in a Video Stream**

[![project intro](http://img.youtube.com/vi/LatP7XUPgIE/0.jpg)](https://youtu.be/LatP7XUPgIE "project intro")

### 3. Project Expectations

**Project Expectations**

For each project in Term 1, keep in mind a few key elements:

* rubric
* code
* writeup
* submission

**Rubric**

Each project comes with a rubric detailing the requirements for passing the project. Project reviewers will check your 
project against the rubric to make sure that it meets specifications.

Before submitting your project, compare your submission against the rubric to make sure you've covered each rubric 
point.

Here is an example of a project rubric:

![example project rubric](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894f8b9_screen-shot-2017-02-03-at-1.39.23-pm/screen-shot-2017-02-03-at-1.39.23-pm.png)

**Code**

Every project in the term includes code that you will write. For some projects we provide code templates, often in a 
Jupyter notebook. For other projects, there are no code templates.

In either case, you'll need to submit your code files as part of the project. Each project has specific instructions 
about what files are required. Make sure that your code is commented and easy for the project reviewers to follow.

For the Jupyter notebooks, sometimes you must run all of the code cells and then export the notebook as an HTML file. 
The notebook will contain instructions for how to do this.

Because running the code can take anywhere from several minutes to a few hours, the HTML file allows project reviewers 
to see your notebook's output without having to run the code.

Even if the project requires submission of the HTML output of your Jupyter notebook, please submit the original Jupyter 
notebook itself, as well.

**Writeup**

All of the projects in Term 1 require a writeup. The writeup is your chance to explain how you approached the project.

It is also an opportunity to show your understanding of key concepts in the program.

We have provided writeup templates for every project so that it is clear what information needs to be in each writeup. 
These templates can be found in each project repository, with the title writeup_template.md.

Your writeup report should explain how you satisfied each requirement in the project rubric.

The write-ups can be turned in either as Markdown files (.md) or PDF files.

**README**

GitHub repositories are a convenient way to organize your projects and display them to the world. A GitHub repository 
also has a README.md file that opens automatically when somebody visits your GitHub repository link.

As a suggestion, the README.md file for each repository can include the following information:

* a list of files contained in the repository with a brief description of each file
* any instructions someone might need for running your code
* an overview of the project

Here is an example of a README file:

![alt text](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5894ff95_screen-shot-2017-02-03-at-2.08.26-pm/screen-shot-2017-02-03-at-2.08.26-pm.png)

If you are unfamiliar with GitHub , Udacity has a brief [GitHub tutorial](http://blog.udacity.com/2015/06/a-beginners-git-github-tutorial.html) 
to get you started. Udacity also provides a more detailed free [course on git and GitHub](https://www.udacity.com/course/how-to-use-git-and-github--ud775).

To learn about README files and Markdown, Udacity provides a free [course on READMEs](https://www.udacity.com/course/writing-readmes--ud777), as well.

GitHub also provides a [tutorial](https://guides.github.com/features/mastering-markdown/) about creating Markdown files.

### 4. Project instructions workspaces

**Workspace Users:**

This **workspace** is designed to be a simple, easy to use environment in which you can code and run the Finding Lane 
Lines project.

**Note:** If you prefer to run the project in your local setup, navigate to the **Project instructions Local Setup** 
lesson.

**Intro**

In this project, you will be writing code to identify lane lines on the road, first in an image, and later in a video 
stream (really just a series of images). To complete this project you will use the tools you learned about in the 
lesson, and build upon them.

Your first goal is to write code including a series of steps (pipeline) that identify and draw the lane lines on a few 
test images. Once you can successfully identify the lines in an image, you can cut and paste your code into the block 
provided to run on a video stream.

You will then refine your pipeline with parameter tuning and by averaging and extrapolating the lines.

Finally, you'll make a brief writeup report. The workspace github repository has a writeup_template.md that can be used 
as a guide.

Have a look at the video clip called "P1_example.mp4" in the repository to see an example of what your final output 
should look like. Two videos are provided for you to run your code on. These are called "solidWhiteRight.mp4" and solidYellowLeft.mp4".

For tips on workspace use, please review the [Workspaces lesson.](https://classroom.udacity.com/nanodegrees/nd013/parts/edf28735-efc1-4b99-8fbb-ba9c432239c8/modules/c773ee09-4a0d-445d-b6af-0739e6653f18/lessons/ec7a3d6d-24a6-4dad-9f5c-aacd29763e0b/concepts/54aad67c-95ab-44c0-b16e-f92dd377137c?contentVersion=2.0.0&contentLocale=en-us)

**Accessing and using the workspace:**

* Go to the workspace node and the project JUPYTER notebook will automatically load
* Complete the project using the instructions in the notebook
* The project repo is already in the workspace. To see other files in the repo click on the JUPYTER icon. This will 
expose the root directory. From there click on the project folder.

**Commit to GitHub**

Students are highly encouraged to commit their project to a GitHub repo. To do this, you must change the upstream of the current repository and add your credentials. We have supplied a bash script to help you do this. Please open up a terminal, navigate to the project repository, and enter: ./set_git.sh, then follow the prompts. This will set the upstream remote to your own repository and add your email and username to the git configuration. At this time we are not configuring passwords, so you will need to enter your username and password for each push. Since credentials are not persistent, it will be necessary to run this script each time you open, refresh, or reset the workspace.

**Things to keep in mind:**

* If you leave your workspace unattended, it will time out and need to be refreshed. Your most recent work will be restored, but the list of open files or any running shell sessions will not be restored.

**Evaluation**

Once you have completed your project, use the [Project Rubric](https://review.udacity.com/#!/rubrics/1967/view) to review the project. If you have covered all of the 
points in the rubric, then you are ready to submit! If you see room for improvement in any category in which you do not 
meet specifications, keep working!

Your project will be evaluated by a Udacity reviewer according to the same [Project Rubric.](https://review.udacity.com/#!/rubrics/1967/view) Your project must "meet 
specifications" in each category in order for your submission to pass.

**Ready to submit your project?**

Make sure your workspace contains at least :

* Jupyter Notebook with your project code
* writeup report (md or pdf file)

Click on the **Submit Project** button and follow the instructions to submit!

**Project Support**

If you are stuck or having difficulties with the project, don't lose hope! Remember to ask (and answer!) questions on 
Knowledge tagged with the project name, and reach out to your fellow students in the #s-t1-p-finding-lane-l channel in 
Slack. We also have a previously recorded project Q&A that you can watch here!

**Share your project success**
Passed your project? Share the good news!

What you’ve accomplished is no small feat. Give yourself a pat on the back and some well-deserved recognition by sharing 
your success with your network.

### 5. Finding lane lines

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Projects/01_find_lane_lines_on_the_road/01_01.PNG)

### 6. Project instructions local setup

**Local Users:**

To get started on the project, download or git clone [the project repository on GitHub](https://github.com/udacity/CarND-LaneLines-P1)
and have a look at the Readme file for detailed instructions on how to get setup with Python and OpenCV and how to 
access the Jupyter Notebook containing the project code. You will need to download, or git clone, this repository in 
order to complete the project. In the next lessons you can find information about setting up the project on your 
computer too.

**Intro**

In this project, you will be writing code to identify lane lines on the road, first in an image, and later in a video 
stream (really just a series of images). To complete this project you will use the tools you learned about in the 
lesson, and build upon them.

Your first goal is to write code including a series of steps (pipeline) that identify and draw the lane lines on a few 
test images. Once you can successfully identify the lines in an image, you can cut and paste your code into the block 
provided to run on a video stream.

You will then refine your pipeline with parameter tuning and by averaging and extrapolating the lines.

Finally, you'll make a brief writeup report. The github repository has a writeup_template.md that can be used as a 
guide.

Have a look at the video clip called "P1_example.mp4" in the repository to see an example of what your final output 
should look like. Two videos are provided for you to run your code on. These are called "solidWhiteRight.mp4" and 
solidYellowLeft.mp4".

**Evaluation**

Once you have completed your project, use the [Project Rubric](https://review.udacity.com/#!/rubrics/322/view) to review the project. If you have covered all of the 
points in the rubric, then you are ready to submit! If you see room for improvement in **any** category in which you do not 
meet specifications, keep working!

Your project will be evaluated by a Udacity reviewer according to the same [Project Rubric.](https://review.udacity.com/#!/rubrics/322/view) 
Your project must "meet specifications" in each category in order for your submission to pass.

**Submission**

**What to include in your submission**

You may submit your project as a zip file or with a link to a github repo. The submission must include two files:

* Jupyter Notebook with your project code
* writeup report (md or pdf file)

**Ready to submit your project?**

Click on the "Submit Project" button and follow the instructions to submit!

**Project Support**

If you are stuck or having difficulties with the project, don't lose hope! Remember to talk to your mentor, ask (and 
answer!) questions on [Knowledge](https://knowledge.udacity.com/) tagged with the project name, and reach out to your fellow students in the #s-t1-p-finding-lane-l 
channel in Slack. We also have a previously recorded project Q&A that you can watch [here](https://youtu.be/hnXkCiM2RSg)!

**Share your project success**

Passed your project? Share the good news!

What you’ve accomplished is no small feat. Give yourself a pat on the back and some well-deserved recognition by sharing 
your success with your network.

### 7. Starter Kit installation

**Starter Kit Installation**

In this term, you'll use Python 3 for programming quizzes, labs, and projects. The following will guide you through 
setting up the programming environment on your local machine.

There are two ways to get up and running:

1. Anaconda
2. Docker

We recommend you first try setting up your environment with Anaconda. It's faster to get up and running and has fewer 
moving parts.

If the Anaconda installation gives you trouble, try Docker instead.

Follow the instructions in [this README.](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md)

Here is a great link for learning more about [Anaconda and Jupyter Notebooks.](https://classroom.udacity.com/courses/ud1111)

### 8. Run some code!

**Run Some Code!**

Now that everything is installed, let's make sure it's working!

1. Clone and navigate to the starter kit test repository.

`# NOTE: This is DIFFERENT from  https://github.com/udacity/CarND-Term1-Starter-Kit.git` <br>
`git clone https://github.com/udacity/CarND-Term1-Starter-Kit-Test.git` <br>
`cd CarND-Term1-Starter-Kit-Test`

2. Launch the Jupyter notebook with Anaconda or Docker. **This notebook is simply to make sure the installed packages 
are working properly**. The instructions for the first project are on the next page.

`# Anaconda` <br>
`source activate carnd-term1 # If currently deactivated, i.e. start of a new terminal session` <br>
`jupyter notebook test.ipynb` <br>

`# Docker` <br>
`docker run -it --rm -p 8888:8888 -v ${pwd}:/src udacity/carnd-term1-starter-kit test.ipynb` <br>
`# OR` <br>
`docker run -it --rm -p 8888:8888 -v `pwd`:/src udacity/carnd-term1-starter-kit test.ipynb` <br>

3. Go to http://localhost:8888/notebooks/test.ipynb in your browser and run all the cells. Everything should execute 
without error.

**Troubleshooting**

**ffmpeg**

**NOTE:** If you don't have ffmpeg installed on your computer you'll have to install it for moviepy to work. If this is 
the case you'll be prompted by an error in the notebook. You can easily install ffmpeg by running the following in a 
code cell in the notebook.

`import imageio` <br>
`imageio.plugins.ffmpeg.download()`

Once it's installed, moviepy should work.

**Docker**

To get the latest version of the [docker image](https://hub.docker.com/r/udacity/carnd-term1-starter-kit/), you may need 
to run:

`docker pull udacity/carnd-term1-starter-kit`

Warning! The image is ~2GB!

### 9. Project: Finding lane lines on the road

![alt text](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Projects/01_find_lane_lines_on_the_road/01_02.PNG)
