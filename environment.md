


## Virtual Environment
```
python3 -m venv .venv
source .venv/bin/activate

rm -rf .venv

which python
deactivate

pip install -r requirements.txt
pip freeze > requirements.txt
```


## Kernel:
```
pip install ipykernel

pip install --upgrade pip

python -m ipykernel install --user --name docqa-kernel --display-name="docQA kernel"

jupyter kernelspec list

jupyter kernelspec uninstall docqa-kernel

```