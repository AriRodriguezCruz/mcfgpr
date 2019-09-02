import ply.lex as lex

# List of token names.   This is always required
tokens = (
   'PROP',
   'TRUE',
   'FALSE',
   'NOT',
   'AND',
   'OR',
   'IMPLY',
   'X',
   'U',
   'F',
   'G',
   'R',
   'LPAREN',
   'RPAREN',
)

# Regular expression rules for simple tokens
t_PROP    = r'[a-zA-Z_][a-zA-Z0-9_]*'
t_TRUE    = r'True'
t_FALSE   = r'False'
t_NOT     = r'Not'
t_AND     = r'And'
t_OR      = r'Or'
t_IMPLY   = r'Imply'
t_X       = r'X'
t_U       = r'U'
t_F       = r'F'
t_G       = r'G'
t_R       = r'R'
t_LPAREN  = r'\('
t_RPAREN  = r'\)'

# A regular expression rule with some action code
def t_NUMBER(t):
    r'\d+'
    t.value = int(t.value)
    return t

# Define a rule so we can track line numbers
def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)

# A string containing ignored characters (spaces and tabs)
t_ignore  = ' \t'

# Error handling rule
def t_error(t):
    print("Illegal character '%s'" % t.value[0])
    t.lexer.skip(1)


# Build the lexer
lexer = lex.lex()