````markdown

# EduQS

This folder contains the preprocessing and evaluation scripts used in **EduQS**.

## Files

- `raw2ques.py`: convert raw data into structured questions
- `ques2prompt.py`: transform questions into prompt format
- `prompt.py`: define prompt templates
- `extractPrompt.py`: extract or organize prompts
- `Img2Text.py`: convert image information into text
- `eval.py`: evaluate model outputs
- `infer&eval.py`: run inference and evaluation together

## Workflow

```text
raw data → raw2ques.py → structured questions → ques2prompt.py / prompt.py → prompt data → inference → eval.py
````

If image information is involved:

```text
image → Img2Text.py → text description → prompt construction
```

## Usage

Run scripts as needed, for example:

```bash
python raw2ques.py
python ques2prompt.py
python infer&eval.py
```

To check available arguments:

```bash
python raw2ques.py --help
python ques2prompt.py --help
python eval.py --help
```

## Notes

* Run the scripts in the correct project environment.
* Modify input/output paths according to your local data location.
* Keep the preprocessing order consistent for reproducibility.

```
```
