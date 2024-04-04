"""
the following code was generated with the help of Claude 3 Opus from
antrhopic.ai

"""

from qtpy.QtCore import QRegularExpression, Qt
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QSyntaxHighlighter,
    QTextCharFormat,
)
from qtpy.QtWidgets import QTextEdit


class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.highlightingRules = []

        # Keyword format
        keywordFormat = QTextCharFormat()
        keywordFormat.setForeground(QColor(147, 226, 241))
        keywordFormat.setFontWeight(QFont.Bold)
        keywords = [
            "if",
            "else",
            "for",
            "while",
            "def",
            "class",
            "try",
            "except",
            "finally",
            "return",
            "break",
            "continue",
            "plt",
            "np",
            "viewer",
            "pd",
        ]
        self.highlightingRules = [
            (QRegularExpression(r"\b" + keyword + r"\b"), keywordFormat)
            for keyword in keywords
        ]

        # String format
        stringFormat = QTextCharFormat()
        stringFormat.setForeground(Qt.darkGreen)
        self.highlightingRules.append(
            (QRegularExpression(r'".*"'), stringFormat)
        )
        self.highlightingRules.append(
            (QRegularExpression(r"'.*'"), stringFormat)
        )

        # Comment format
        commentFormat = QTextCharFormat()
        commentFormat.setForeground(Qt.darkGray)
        self.highlightingRules.append(
            (QRegularExpression(r"#[^\n]*"), commentFormat)
        )

    def highlightBlock(self, text):
        for pattern, format in self.highlightingRules:
            expression = QRegularExpression(pattern)
            iterator = expression.globalMatch(text)
            while iterator.hasNext():
                match = iterator.next()
                self.setFormat(
                    match.capturedStart(), match.capturedLength(), format
                )


class CodeEditor(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set to system default fixed-width font
        fixed_font = QFont("Courier new")
        self.setFont(fixed_font)

        # Set tab stop to be 4 spaces
        self.setTabStopDistance(
            QFontMetrics(fixed_font).horizontalAdvance(" ") * 4
        )

        # Set up syntax highlighting
        self.highlighter = PythonHighlighter(self.document())
