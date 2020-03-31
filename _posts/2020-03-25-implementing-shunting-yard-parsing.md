---
layout: post
title: Implementing shunting-yard parsing in Python  
share: false
tags: [parsing, shunting-yard, expression tree, binary tree, arithmetic expression]
---
  
Consider the problem of parsing an arithmetic expression, such as `4*(1+6)/3`, into a binary expression tree. The problem would be quite easy with postfix notation (also known as reverse Polish notation): we could parse `416+*3/`simply by reading the expression left to right, pushing each operand (number) on a stack and, upon encountering an operator, building a node with the operator and two operands from the stack and pushing the node back on the stack. Eventually, a root node of the expression tree will be the only remaining element on the stack. The infix notation forces us to deal with the nastiness of operator precedence and parentheses. Let us break this problem down and focus first on parentheses alone, then operator precedence alone, and finally both of them.  
  
An expression tree for an arithmetic expressions can be implemented as follows:  
  
```python
from dataclasses import dataclass  # if python < 3.7
from typing import Optional, List
from operator import add, sub, mul, truediv

@dataclass
class Node:
    symbol: str
    left: Optional['Node']
    right: Optional['Node']

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

@dataclass
class Tree:
    root: Node

    @classmethod
    def _tokenize(cls, text: str) -> List[str]:
        prev = ''
        tokenized = []
        for char in text:
            if prev.isdigit() and char.isdigit():
                tokenized.append(tokenized.pop() + char)
            else:
                tokenized.append(char)
            prev = char
        return tokenized

    @classmethod
    def build(cls, text: str) -> 'Tree':
        raise NotImplementedError

    def evaluate(self, node: Optional[Node] = None):
        OPS = {
            '+': add,
            '-': sub,
            '*': mul,
            '/': truediv
        }
        node = node or self.root
        if node.is_leaf():
            return int(node.symbol)
        else:
            op = OPS[node.symbol]
            return op(self.evaluate(node.left), self.evaluate(node.right))
```

`tokenize` is needed to have multi-digit numbers such as `11` as single operands. Evaluation of an expression tree boils down to recursively evaluating both subtrees of a tree and applying an operator to the result.
  
## Parsing with parentheses

Expressions with evaluation order determined unambiguously such as `(((2+3)/2)*3)` can be parsed with a single stack for both the operators and operands. We just ignore opening parentheses (`(`) and, upon encountering `)` we pull last tree elements from the stack --- which are guaranteed to be an operand, an operator and an operand --- and push a corresponding `node` back to the stack.

```python
    @classmethod
    def build(cls, text: str) -> 'Tree':
        stack: List[Node] = []
        for char in cls._tokenize(text):
            if char.isdigit():
                stack.append(Node(symbol=char, left=None, right=None))
            elif char == ')':
                right = stack.pop()
                op = stack.pop()
                left = stack.pop()
                stack.append(Node(symbol=op, left=left, right=right))
            elif char in '+-*/':
                stack.append(char)
        return cls(root=stack.pop())
```

## Parsing with operator precedence

Expression such as `2+3*5` (without parentheses) can be parsed by relying on operator precedence, i.e. the fact that we should first evaluate division and multiplication and only then addition and subtraction. We can implement this with two separate stacks (for operators and operands). If we encounter `+` or `-` and there is `*` or `\` on the stack, we should push `*` or `/` from the operator stack, build a `Node` with two most recent operands and push the `node` onto the operand stack and the first operator (`+` or `-`) onto the operator stack.

```python
    @classmethod
    def build(cls, text: str) -> 'Tree':
        operator_stack: List[str] = []
        operand_stack: List[Node] = []
        for char in cls._tokenize(text):
            print(operator_stack, operand_stack)
            if char.isdigit():
                operand_stack.append(Node(symbol=char, left=None, right=None))
            elif char in '+-' and len(operator_stack) > 0 and operator_stack[-1] in '*/':
                right = operand_stack.pop()
                op = operator_stack.pop()
                left = operand_stack.pop()
                operand_stack.append(Node(symbol=op, left=left, right=right))
                operator_stack.append(char)
            else:
                operator_stack.append(char)
        while len(operator_stack) > 0:
            right = operand_stack.pop()
            op = operator_stack.pop()
            left = operand_stack.pop()
            operand_stack.append(Node(symbol=op, left=left, right=right))
        return cls(root=operand_stack.pop())
```

## Parsing with both operator precedence and parentheses

Finally, support for both operator precedence and parentheses can be pushing `(` onto the operator stack and, upon encountering `)`, applying all operator from the stack until `)` again is on the top of the operator stack (and can be discarded).

```python
    @classmethod
    def build(cls, text: str) -> 'Tree':
        operator_stack: List[str] = []
        operand_stack: List[Node] = []
        for char in cls._tokenize(text):
            print(operator_stack, operand_stack)
            if char.isdigit():
                operand_stack.append(Node(symbol=char, left=None, right=None))
            elif char in '+-' and len(operator_stack) > 0 and operator_stack[-1] in '*/':
                right = operand_stack.pop()
                op = operator_stack.pop()
                left = operand_stack.pop()
                operand_stack.append(Node(symbol=op, left=left, right=right))
                operator_stack.append(char)
            elif char == ')':
                while len(operator_stack) > 0 and operator_stack[-1] != '(':
                    right = operand_stack.pop()
                    op = operator_stack.pop()
                    left = operand_stack.pop()
                    operand_stack.append(Node(symbol=op, left=left, right=right))
                operator_stack.pop()
            else:
                operator_stack.append(char)
        while len(operator_stack) > 0:
            right = operand_stack.pop()
            op = operator_stack.pop()
            left = operand_stack.pop()
            operand_stack.append(Node(symbol=op, left=left, right=right))
        return cls(root=operand_stack.pop())
```
It works as expected:
```
tree = Tree.build('(2+3)*2+7*3')  
assert tree.evaluate() == 31
```

The final code snippet is essentially Dijkstra's [shunting-yard algorithm](https://en.wikipedia.org/wiki/Shunting-yard_algorithm).

One final remark: parsing and evaluating can be done in one gone, without building a whole expression tree. You could just evaluate an operator and push the result on the stack instead of building a `Node` and pushing it on the stack.