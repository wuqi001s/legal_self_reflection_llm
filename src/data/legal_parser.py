import re
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LegalClause:
    """Legal clause structure with metadata"""
    id: str
    title: str
    content: str
    level: str
    parent_id: Optional[str]
    chapter: str
    section: str
    metadata: Dict[str, Any]


class ASTLegalParser:
    """AST-based legal document parser for hierarchical parsing"""
    
    LEVEL_PATTERNS = {
        '编': r'^第[一二三四五六七八九十百千]+编\s+(.+)',
        '章': r'^第[一二三四五六七八九十百千]+章\s+(.+)',
        '节': r'^第[一二三四五六七八九十百千]+节\s+(.+)',
        '条': r'^第[零一二三四五六七八九十百千]+条\s+(.+)',
    }
    
    CHINESE_NUMBERS = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '百': 100, '千': 1000
    }
    
    def __init__(self):
        self.current_book = ""
        self.current_chapter = ""
        self.current_section = ""
        
    def parse_docx(self, docx_path: str) -> List[LegalClause]:
        """Parse Civil Code from docx file"""
        from docx import Document
        
        doc = Document(docx_path)
        clauses = []
        content_lines = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                content_lines.append(text)
        
        current_book = ""
        current_chapter = ""
        current_section = ""
        clause_counter = 0
        
        i = 0
        while i < len(content_lines):
            line = content_lines[i]
            
            if re.match(r'^第[一二三四五六七八九十百千]+编\s+', line):
                current_book = line
                current_chapter = ""
                current_section = ""
            elif re.match(r'^第[一二三四五六七八九十百千]+章\s+', line):
                current_chapter = line
                current_section = ""
            elif re.match(r'^第[一二三四五六七八九十百千]+节\s+', line):
                current_section = line
            elif re.match(r'^第[零一二三四五六七八九十百千]+条\s+', line):
                clause_counter += 1
                match = re.match(r'^(第[零一二三四五六七八九十百千]+条)\s+(.+)', line)
                clause_title = match.group(1) if match else line[:20]
                clause_content = match.group(2) if match else line
                clause = LegalClause(
                    id=f"clause_{clause_counter}",
                    title=clause_title,
                    content=clause_content,
                    level="条",
                    parent_id=f"chapter_{current_chapter}" if current_chapter else None,
                    chapter=current_chapter,
                    section=current_section,
                    metadata={"line_num": i}
                )
                clauses.append(clause)
            
            i += 1
                
        return clauses
    
    def parse_file(self, file_path: str) -> List[LegalClause]:
        """Parse a legal document file into hierarchical clauses"""
        if file_path.endswith('.docx'):
            return self.parse_docx(file_path)
            
        clauses = []
        content = Path(file_path).read_text(encoding='utf-8')
        
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            clause = self._parse_line(line, i)
            if clause:
                clauses.append(clause)
                
        return clauses
    
    def _parse_line(self, line: str, line_num: int) -> Optional[LegalClause]:
        """Parse a single line and determine its hierarchy level"""
        for level, pattern in self.LEVEL_PATTERNS.items():
            match = re.match(pattern, line)
            if match:
                title = match.group(1) if match.groups() else ""
                
                if level == '编':
                    self.current_book = title
                elif level == '章':
                    self.current_chapter = title
                elif level == '节':
                    self.current_section = title
                    
                return LegalClause(
                    id=f"clause_{line_num}",
                    title=title,
                    content=line,
                    level=level,
                    parent_id=self._get_parent_id(level),
                    chapter=self.current_chapter,
                    section=self.current_section,
                    metadata={"line_num": line_num}
                )
        return None
    
    def _get_parent_id(self, level: str) -> Optional[str]:
        """Get parent ID based on current hierarchy"""
        level_order = ['编', '章', '节', '条']
        current_idx = level_order.index(level) if level in level_order else -1
        
        for parent_level in reversed(level_order[:current_idx]):
            if parent_level == '编' and self.current_book:
                return "book_root"
            elif parent_level == '章' and self.current_chapter:
                return f"chapter_{self.current_chapter}"
            elif parent_level == '节' and self.current_section:
                return f"section_{self.current_section}"
        return None
    
    def save_clauses(self, clauses: List[LegalClause], output_path: str):
        """Save parsed clauses to JSON"""
        data = [ {
            "id": c.id,
            "title": c.title,
            "content": c.content,
            "level": c.level,
            "parent_id": c.parent_id,
            "chapter": c.chapter,
            "section": c.section,
            "metadata": c.metadata
        } for c in clauses ]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


class LawDataLoader:
    """Load and process lawzhidao data for QLoRA"""
    
    @staticmethod
    def load_best_qa(csv_path: str, max_samples: Optional[int] = None) -> List[Dict]:
        """Load QA pairs where is_best=1"""
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('is_best') == '1':
                    data.append({
                        'title': row.get('title', ''),
                        'question': row.get('question', ''),
                        'reply': row.get('reply', '')
                    })
        
        if max_samples:
            data = data[:max_samples]
        return data
    
    @staticmethod
    def save_qa_pairs(qa_pairs: List[Dict], output_path: str):
        """Save QA pairs to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(qa_pairs, f, ensure_ascii=False, indent=2)


class LegalDataGenerator:
    """Generate legal Q&A data from parsed clauses"""
    
    def __init__(self, model=None):
        self.model = model
        
    def generate_qa_pairs(self, clauses: List[LegalClause], num_pairs: int = 100) -> List[Dict]:
        """Generate QA pairs from legal clauses"""
        qa_pairs = []
        
        for clause in clauses[:num_pairs]:
            qa = self._generate_single_qa(clause)
            if qa:
                qa_pairs.append(qa)
                
        return qa_pairs
    
    def _generate_single_qa(self, clause: LegalClause) -> Optional[Dict]:
        """Generate a single QA pair from a clause"""
        if not self.model:
            return {
                "question": f"请解释{clause.title}的内容",
                "answer": clause.content,
                "metadata": {
                    "level": clause.level,
                    "chapter": clause.chapter,
                    "section": clause.section
                }
            }
        return None
    
    def save_qa_pairs(self, qa_pairs: List[Dict], output_path: str):
        """Save QA pairs to JSONL format"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + '\n')
