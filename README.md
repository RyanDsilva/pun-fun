# CNIT 519 Project

### Setup

- Have Python 3.10 and pipenv install

1. Create virtual environment

```sh
pipenv --python 3.10
```

2. Activate virtual environment
3. Install dependencies

```sh
pipenv install
```

4. Run project

```sh
python3 main.py
```

> If you change any code in the submodules, you need to run `pipenv install` for the changes to reflect

### Example Outputs (Project 2)

```md
Sentence: He didn't attend the funeral because he was not a mourning person
Source word: mourning
Predicted target: morning
Predicted sentence: He didn't attend the funeral because he was not a morning person

======================================================================================

**Definitions**

mourning: state of sorrow over the death or departure of a loved one
morning: the time period between dawn and noon

======================================================================================

Sense similarity between mourning and morning is 0.07692307692307693
Sound similarity between mourning and morning is 100
```
