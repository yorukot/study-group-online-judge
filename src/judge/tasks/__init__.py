from judge.tasks.base import Task
from judge.tasks.lab1 import Lab1
from judge.tasks.lab2 import Lab2
from judge.tasks.lab4 import Lab4
from judge.tasks.lab5 import Lab5

TASKS: dict[str, Task] = {
    Lab1.id: Lab1(),
    Lab2.id: Lab2(),
    Lab4.id: Lab4(),
    Lab5.id: Lab5(),
}
