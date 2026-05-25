"""Test package — declared explicitly so pytest treats tests/ as a
package and assigns fully-qualified module names (e.g.,
`tests.test_foo` rather than rootdir-relative `test_foo`). Without
this, two test files of the same basename in different directories
collide and one is silently dropped during collection (see
~/.claude/rules/lessons.md \"Module-name collision hides tests\").
"""
