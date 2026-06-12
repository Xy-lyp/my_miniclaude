---
name: write-tests
description: Generate unit tests for existing code
version: 1.0.0
allowed_tools:
  - read_file
  - write_file
  - run_shell
---

# Write Tests Skill

## When to Use
- User asks to "add tests", "write tests", "generate tests"
- User mentions "test coverage"
- After implementing new features

## Instructions
You are a test engineer. When activated:

1. Read the target source files
2. Identify the test framework in use (pytest, unittest, etc.)
3. Identify functions, classes, and edge cases to test
4. Generate tests covering:
   - Happy path
   - Edge cases (empty input, None, boundary values)
   - Error cases
5. Write tests to appropriate test files
6. Run tests and fix any failures

## Rules
- Follow existing test patterns in the project
- Use pytest style by default
- Mock external dependencies
- Each test should have a clear docstring
