# Lesson II: Workspaces

### 1. Introduction to workspaces

**Introduction**

Many projects and select quizzes will be accessed via Workspaces. These workspaces streamline environment setup, 
simplify project submission, and can be enabled with GPU support. All workspaces are Linux-based and can be interfaced 
via a shell (BASH). Some workspace interfaces are direct from the shell, others run a JUPYTER Notebook server and 
interaction is mainly through the JUPYTER notebook interface. To learn more about JUPYTER, please visit [the official 
site of the JUPYTER project](http://jupyter.org/). Shell workspaces operate in the same manner as Linux terminals/shells.

If you are unfamiliar with Jupyter Notebooks, check out this quick Udacity primer on [Anaconda and Jupyter Notebooks](
https://classroom.udacity.com/courses/ud1111).

**Which quizzes and projects are accessible through workspaces?**

If a quiz or project is accessible through a workspace there will be a workspace evident for use. For projects the 
workspace will be towards the end of the project lesson. For quizzes the workspace will either be in the node where the 
quiz is introduced or nearby (usually directly after). If that is not the case for a particular quiz or project, then we 
currently don't support workspaces for that quiz or project.

**Example workspaces**

Example workspaces can be found at the end of this lesson. Please note that some functionality, such as enable GPU and 
project submission, are not present in the example workspaces. Feel free to experiment and familiarize yourself with 
workspaces. Have fun!

**JUPYTER workspace example**

![jupyter workspace example](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/January/5a501a04_workspaces-jupyter/workspaces-jupyter.png)

**Shell workspace example**

![shell workspace example](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5b046b1c_workspace-paste-ex/workspace-paste-ex.png)

**Important Notes:**

* Workspaces sessions are connections from your browser to a remote server. Udacity **Workspaces with GPU support** are 
available for some projects as an alternative to manually configuring your own remote server with GPU support. Each 
student has a limited number of GPU hours allocated on the servers. The current allocation is 50 hours, significantly 
more than completing the projects is expected to take.

* There is currently no limit on the number of Workspace hours when GPU mode is disabled.

* Workspace data stored in the user's home folder is preserved between sessions (and can be reset as needed, e.g., to 
get project updates).

* **Only 3 gigabytes of data can be stored in the home folder**.

* Workspace sessions are preserved if your connection drops or your browser window is closed, simply return to the 
classroom and re-open the workspace page; however, workspace sessions are automatically terminated after a period of 
inactivity. This will prevent you from leaving a session connection open and burning through your time allocation. 
(See the section on active connections below.)

* The kernel state is preserved as long as the notebook session remains open, but it is not preserved if the session is 
closed. If you exit the notebook for more than half an hour and the session is closed, you will need to re-run any 
previously-run cells before continuing.

**Reporting Issues**

If you find any issues or bugs with the materials in this lesson, or you have suggestions for improvement, we would 
appreciate it if you would take the time to post them [here](https://github.com/udacity/sdc-issue-reports/issues).

### 2. Using workspaces

**Using Workspaces**

This lesson will briefly introduce the Workspaces interface for JUPYTER Notebooks. The same rules apply when using a 
shell interface since a JUPYTER workspace is just a workspace that is serving a JUPYTER notebook.

When the workspace opens, you'll see the normal Jupyter file browser. From this interface you can open a notebook file, 
start a remote terminal session, enable the GPU, submit your project, or reset the workspace data, and more. Clicking 
the three bars in the top left corner above the Jupyter logo will toggle hiding the classroom lessons sidebar.

**NOTE: You can always return to the file browser page from anywhere else in the workspace by clicking the Jupyter logo 
in the top left corner.**

**Opening a notebook**

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b229e7c_workspaces-notebook/workspaces-notebook.png)

Clicking the name of a notebook (*.ipynb) file in the file list will open a standard Jupyter notebook view of the 
project. The notebook session will remain open as long as you are active, and will be automatically terminated after 30 
minutes of inactivity.

You can exit a notebook by clicking on the Jupyter logo in the top left corner.

**NOTE: Notebooks continue to run in the background unless they are stopped. IF GPU MODE IS ACTIVE, IT WILL REMAIN 
ACTIVE AFTER CLOSING OR STOPPING A NOTEBOOK. YOU CAN ONLY STOP GPU MODE WITH THE GPU TOGGLE BUTTON. (See next section.)**

### 3. GPU workspaces

**Enabling GPU Mode**

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b229efb_workspaces-gpu/workspaces-gpu.png)

GPU Workspaces can also be run without time restrictions when the GPU mode is disabled. The "Enable"/"Disable" button 
(circled in red in the image) can be used to toggle GPU mode. **NOTE: Toggling GPU support may switch the physical server 
your session connects to, which can cause data loss UNLESS YOU CLICK THE SAVE BUTTON BEFORE TOGGLING GPU SUPPORT**.

**ALWAYS SAVE YOUR CHANGES BEFORE TOGGLING GPU SUPPORT**.

**Keeping Your Session Active**

Workspaces automatically disconnect after 30 minutes of inactivity.

### 4. Submitting projects

**Submitting Projects**

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b229fba_workspaces-submit/workspaces-submit.png)

Some workspaces are able to directly submit projects on your behalf (i.e., you do not need to manually submit the 
project in the classroom). To submit your project, simply click the "Submit Project" button (circled in red in the above
image).

If you do not see the "Submit Project" button, then project submission is not enabled for that workspace. You will need 
to manually download your project files and submit them in the classroom.

**NOTE: YOU MUST ENSURE THAT YOUR SUBMISSION INCLUDES ALL REQUIRED FILES BEFORE SUBMITTING -- INCLUDING ANY FILE 
CONVERSIONS (e.g., from ipynb to HTML)**

### 5. Terminals

**Opening a terminal**

Jupyter workspaces support several views, including the file browser and notebook view already covered, as well as shell
terminals. To open a terminal shell, click the "New" menu button at the top right of the file browser view and select 
"Terminal".

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b22a049_workspaces-new/workspaces-new.png)

Terminals provide a full Bash shell that you can use to install or update software packages, fetch updates from github 
repositories, or run any other terminal commands. As with the notebook view, you can return to the file browser view by 
clicking on the Jupyter logo at the top left corner of the window.

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b22a084_workspaces-terminal/workspaces-terminal.png)

**NOTE: Your data & changes are persistent across workspace sessions. Any changes you make will need to be repeated if you 
later reset your workspace data.**

### 6. Resetting data

**Resetting Data**

![alt text](https://s3.amazonaws.com/video.udacity-data.com/topher/2018/June/5b22a14d_workspaces-menu/workspaces-menu.png)

The "Menu" button in the bottom left corner provides support for resetting your Workspaces. The "Refresh Workspace" 
button will refresh your session, which has no effect on the changes you've made in the workspace.

The "Reset Data" button discards all changes and restores a clean copy of the workspace. Clicking the button will open a 
dialog that requires you to type "Reset data" in a confirmation dialog. **ALL OF YOUR DATA WILL BE LOST**.

Resetting should only be required if Udacity makes changes to the project and you can't get them via git pull, or if you 
destroy the contents of the workspace. If you do need to reset your data, you are strongly encouraged to download a copy 
of your work from the file interface before clicking Reset Data.

### 7. Example Jupyter workspace

![quiz](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/02_01.PNG)

### 8. Example terminal workspace

![quiz](https://raw.githubusercontent.com/swoldetsadick/sdce/master/Lessons/images/02_02.PNG)
