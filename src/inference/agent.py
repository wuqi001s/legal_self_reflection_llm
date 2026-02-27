from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class ActionType(Enum):
    """Available legal agent actions"""
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    ANALYZE = "analyze"
    REFLECT = "reflect"
    FINALIZE = "finalize"
    REFUSE = "refuse"


@dataclass
class AgentAction:
    """Agent action with ReAct format"""
    thought: str
    action: ActionType
    action_input: Dict[str, Any]
    observation: str = ""


@dataclass
class AgentState:
    """Current state of the legal agent"""
    query: str
    actions: List[AgentAction] = field(default_factory=list)
    retrieved_docs: List[Any] = field(default_factory=list)
    reflection_result: Optional[Dict] = field(default_factory=dict)
    final_answer: str = ""
    should_refuse: bool = False
    refusal_reason: str = ""
    iteration: int = 0


class SelfReflection:
    """
    Self-Reflection mechanism for verifying retrieval completeness
    Triggers secondary/tertiary search on information gaps
    """
    
    def __init__(self, llm=None):
        self.llm = llm
        
    def reflect(self, query: str, retrieved_docs: List[Any]) -> Dict[str, Any]:
        """
        Evaluate if retrieved documents are sufficient
        Returns reflection result with gap analysis
        """
        if not self.llm:
            return self._template_reflection(query, retrieved_docs)
            
        prompt = self._build_reflection_prompt(query, retrieved_docs)
        response = self.llm.generate(prompt)
        
        try:
            return json.loads(response)
        except:
            return self._template_reflection(query, retrieved_docs)
    
    def _build_reflection_prompt(self, query: str, retrieved_docs: List[Any]) -> str:
        """Build reflection prompt"""
        doc_summary = "\n".join([
            f"{i+1}. {doc.content[:200]}..." 
            for i, doc in enumerate(retrieved_docs)
        ])
        
        return f"""请评估以下检索结果是否能完整回答用户问题。

用户问题: {query}

检索到的文档:
{doc_summary}

请判断:
1. 是否有足够的法律依据
2. 是否存在信息缺口
3. 需要补充检索的关键词

输出JSON格式:
{{
    "is_sufficient": true/false,
    "gaps": ["缺口描述"],
    "additional_keywords": ["关键词1", "关键词2"],
    "confidence": 0.0-1.0
}}"""
    
    def _template_reflection(self, query: str, retrieved_docs: List[Any]) -> Dict:
        """Template-based reflection"""
        has_docs = len(retrieved_docs) > 0
        return {
            "is_sufficient": has_docs,
            "gaps": [] if has_docs else ["缺少法律依据"],
            "additional_keywords": [],
            "confidence": 0.8 if has_docs else 0.2
        }


class LegalAgent:
    """
    Agentic RAG with Self-Reflection
    Dynamic tool calling with legal domain knowledge
    """
    
    def __init__(
        self,
        retriever: Any,
        llm: Any,
        reranker: Any = None,
        max_iterations: int = 3,
        reflection_threshold: float = 0.7
    ):
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.max_iterations = max_iterations
        self.reflection_threshold = reflection_threshold
        self.reflection = SelfReflection(llm)
        
    def run(self, query: str) -> AgentState:
        """Execute agentic RAG loop with self-reflection"""
        state = AgentState(query=query)
        
        state = self._check_safety(query, state)
        if state.should_refuse:
            return state
            
        for iteration in range(self.max_iterations):
            state.iteration = iteration + 1
            
            state = self._retrieve_and_reflect(state)
            
            if not state.reflection_result.get("is_sufficient", True):
                if iteration >= self.max_iterations - 1:
                    break
                continue
            else:
                break
                
        state = self._generate_answer(state)
        
        return state
    
    def _check_safety(self, query: str, state: AgentState) -> AgentState:
        """Check if query is safe to answer"""
        unsafe_keywords = ["杀人", "抢劫", "贩毒", "伪造证据", "恐怖袭击"]
        
        for keyword in unsafe_keywords:
            if keyword in query:
                state.should_refuse = True
                state.refusal_reason = "您的问题涉及违法内容，我无法提供相关帮助。"
                state.final_answer = state.refusal_reason
                return state
                
        return state
    
    def _retrieve_and_reflect(self, state: AgentState) -> AgentState:
        """Retrieve documents and apply self-reflection"""
        query = state.query
        
        action = AgentAction(
            thought="我需要检索相关法律条文",
            action=ActionType.RETRIEVE,
            action_input={"query": query, "top_k": 10}
        )
        
        retrieved = self.retriever.retrieve(query, top_k=10)
        action.observation = f"检索到{len(retrieved)}条相关文档"
        
        state.retrieved_docs = retrieved
        state.actions.append(action)
        
        reflection_result = self.reflection.reflect(query, retrieved)
        state.reflection_result = reflection_result
        
        if reflection_result.get("additional_keywords"):
            additional_query = " ".join(reflection_result["additional_keywords"])
            additional_docs = self.retriever.retrieve(additional_query, top_k=5)
            state.retrieved_docs.extend(additional_docs)
            
        return state
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate final answer from retrieved documents"""
        if not state.retrieved_docs:
            state.final_answer = "抱歉，我未能检索到相关的法律条文来回答您的问题。"
            return state
            
        context = "\n\n".join([
            f"【{i+1}】{doc.content}" 
            for i, doc in enumerate(state.retrieved_docs[:5])
        ])
        
        if not self.llm:
            state.final_answer = f"根据检索到的法律条文：\n{context[:500]}..."
            return state
            
        prompt = f"""你是一个专业的法律助手。请根据以下法律条文回答用户问题。

用户问题: {state.query}

相关法律条文:
{context}

请给出专业、准确的回答。回答时应当：
1. 引用具体的法律条文
2. 使用法言法语
3. 逻辑清晰，层次分明

回答:"""
        
        answer = self.llm.generate(prompt)
        state.final_answer = answer
        
        return state
    
    def get_trace(self, state: AgentState) -> List[Dict]:
        """Get execution trace for debugging"""
        return [
            {
                "thought": a.thought,
                "action": a.action.value,
                "input": a.action_input,
                "observation": a.observation
            }
            for a in state.actions
        ]
