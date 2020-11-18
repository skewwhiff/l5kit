import os

os.system('pip uninstall typing -y')

# Hold back nbconvert to avoid https://github.com/jupyter/nbconvert/issues/1384
os.system('git clone https://github.com/skewwhiff/l5kit.git')
os.chdir('/kaggle/working/l5kit/l5kit')
os.system('pip install .  --ignore-installed --target=/kaggle/working')
os.chdir('/kaggle/working')
os.system('rm -rf /kaggle/working/l5kit')
