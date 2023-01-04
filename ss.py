import re

word_input = "가2자"

print(re.sub("[^가-힣]", "", word_input))