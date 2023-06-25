封闭域联合模型
- https://shiina18.github.io/machine%20learning/2021/07/29/tplinker/
- https://zhuanlan.zhihu.com/p/498089575



- 苏GlobalPointer
    - n(n+1)/2个span，也就是n*n序列矩阵的上三角，每个位置为该span为实体的得分
    - 具体计算：向量序列分别映射出一个q向量序列，和一个k向量序列，两者内积得到矩阵；相对位置编码


- SpanBert：（解决实体嵌套和关系嵌套问题）
    - span_representation: funsion_funciton + span长度编码
    - span_classifier: 输入为span_representation拼接[CLS]token， softmax识别实体类型
    - span_filtering: 剔除过长的span和非实体span
    - relation_classification: 
        - 两两span关系分类
        - 拼接span_representation，两span直接context的max_pooling表示
        - sigmoid判断是否存在该种关系（解决嵌套关系）


- ETL_span


-  苏CasRel(2020): 参数共享的联合模型
    - Cascade Relational Triple Extraction，级联式
    - 解决嵌套问题，先抽取subject，再同时抽取predicate 和 object，思路同ETL
    - 不仅解决实体嵌套问题，同时解决关系嵌套问题
    - 不同于ETL，用一个序列标注把所有predict和object同时抽出，而是对每个关系类型都要单独做对应的object抽取，N个关系，就有2N个指针序列（分别标识start和end位置）
    - 编码层为BERT，第一个解码层输出所有subject_span，然后对每个subject_span的编码做一个映射（因为每个span长度不同）然后与sequence_embedding融合（pooling或者其他），作为输入给到第二个解码层，其包含2N个序列标注任务，然后对结果做解析。

    - 问题：
        - 计算效率低，如果关系类型很多，存在大量冗余计算，训练时，每条text是采样一个subject训练的，预测时由于单条text抽取subject数量不固定，所以batch_size一般只能取1。
        - exposure bias：训练时用gold subject作为输入训练抽取predict和object
        - 误差传播：级联模型的通病了


- TPLinker(2020)：联合解码的联合模型
    - Token Pair Linking
    - 构造1+2N的全局信息矩阵（N为关系数量）
    - 方法：
        使用一个矩阵抽取所有的实体，类似global pointer
        N个关系矩阵的表现形式：每个关系矩阵预测转化为：n*(n+1)/2长度的两个序列分别抽取SH_to_OH(标记为1) 和 ST_to_OT（标记为2）
        handshaking tagger解码

    - 问题：
        类似CasrRel关系类型多的时候，冗余计算太多，十分稀疏



- 陈丹琦PURE（2021）：pipeline方法
    - Princeton University Relation Extraction
    - 实体抽取和关系分类使用两个encoder，关注不同的语义信息
    - 实体信息（包括实体边界和实体类型信息）对关系分类很重要，应该尽早融合到关系是识别，而不是在编码层输出之后再融合
    - 方法：
        抽实体：subject和object，依然是token pair的形式，span表示采用将头token和尾token以及token对的距离embedding三者拼接起来。
        抽关系：输入：使用标识符插入句子中实体span的起止位置前后（引入实体信息），然后将编码层输出的标识符token embedding拼接起来作为这个实体对的表示，然后进行关系分类



- NER漏标问题
    - 使用span level抽取方式
    - 对负样本进行采样，减少漏标的实体被作为负样本进行干扰训练



## 事件抽取
- 问答的形式

- seq2seq
    - TEXT2EVENT: Controllable Sequence-to-Structure Generation for End-to-end Event Extraction
        - 设计了一种结构化事件表示形式，对每个事件类型，使用括号等分割触发词，事件类型，事件角色要素
        - 基于字典树的约束解码，在解码的时候在字典树上游走，遇到需要预测事件类型或者角色要素的时候就从事件schema字典树种选择，遇到需要预测span内容的时候就从原文选token
        - 为了保证效果，预训练阶段构建<sentence, substructure>标注语料，让生成模型根据sentence预测substructure
        

        - 优势：
            标注效率高，不需要具体到实体的具体位置
            迁移学习效果好


- 