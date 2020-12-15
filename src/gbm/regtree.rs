use crate::errors::*;
use crate::fvec::FVec;
use crate::model_reader::ModelReader;
use std::f32;

#[derive(Clone, Copy)]
struct Param {
    /// number of start root
    num_roots: i32,
    /// total number of nodes
    num_nodes: i32,
    /// number of deleted nodes
    num_deleted: i32,
    /// maximum depth, this is a statistics of the tree
    max_depth: i32,
    /// number of features used for tree construction
    num_feature: i32,
    /// leaf vector size, used for vector tree used to store more than one dimensional information in tree
    size_leaf_vector: i32,
}

impl Param {
    fn read_from<T: ModelReader>(reader: &mut T) -> Result<Param> {
        let (num_roots, num_nodes, num_deleted, max_depth, num_feature, size_leaf_vector) = (
            reader.read_i32_le()?,
            reader.read_i32_le()?,
            reader.read_i32_le()?,
            reader.read_i32_le()?,
            reader.read_i32_le()?,
            reader.read_i32_le()?,
        );

        let mut reserved = [0i32; 31];
        reader.read_to_i32_buffer(&mut reserved)?;

        return Ok(Param {
            num_roots,
            num_nodes,
            num_deleted,
            max_depth,
            num_feature,
            size_leaf_vector,
        });
    }
}

#[derive(Clone, Copy)]
enum LeafOrSplit {
    LeafValue(f32),
    Split {
        /// pointer to right
        cleft: i32,
        /// pointer to right
        cright: i32,
        /// split feature index, left split or right split depends on the highest bit
        split_cond: f32,
        default_next: i32,
        split_index: i32,
    },
}

#[derive(Clone, Copy)]
struct Node {
    /// pointer to parent, highest bit is used to indicate whether it's a left child or not
    parent: i32,
    leaf_or_split: LeafOrSplit,
}

impl Node {
    fn is_default_left(sindex: i32) -> bool {
        return (sindex >> 31) != 0;
    }

    fn decode_split_index(sindex: i32) -> i32 {
        return ((sindex as i64) & ((1i64 << 31) - 1i64)) as i32;
    }

    fn read_from<T: ModelReader>(reader: &mut T) -> Result<Node> {
        let (parent, cleft, cright, sindex) = (
            reader.read_i32_le()?,
            reader.read_i32_le()?,
            reader.read_i32_le()?,
            reader.read_i32_le()?,
        );

        let split_index = Node::decode_split_index(sindex);
        let default_next = if Node::is_default_left(sindex) {
            cleft
        } else {
            cright
        };

        let is_leaf = cleft == -1;

        let leaf_or_split = if is_leaf {
            LeafOrSplit::LeafValue(reader.read_f32_le()?)
        } else {
            LeafOrSplit::Split {
                cleft,
                cright,
                split_cond: reader.read_f32_le()?,
                default_next,
                split_index,
            }
        };

        return Ok(Node {
            parent,
            leaf_or_split,
        });
    }

    fn next<F: FVec>(&self, feat: &F) -> Option<usize> {
        return match self.leaf_or_split {
            LeafOrSplit::LeafValue(_) => None,
            LeafOrSplit::Split {
                cleft,
                cright,
                split_cond,
                default_next,
                split_index,
            } => match feat.fvalue(split_index as usize, 1) {
                None => return Some(default_next as usize),
                Some(fvalue) => {
                    if fvalue < split_cond {
                        Some(cleft as usize)
                    } else {
                        Some(cright as usize)
                    }
                }
            },
        };
    }
}

#[derive(Clone, Copy)]
struct RTreeNodeStat {
    /// loss chg caused by current split
    loss_chg: f32,
    /// sum of hessian values, used to measure coverage of data
    sum_hess: f32,
    /// weight of current node
    base_weight: f32,
    /// number of child that is leaf node known up to now
    leaf_child_cnt: i32,
}

impl RTreeNodeStat {
    fn read_from<T: ModelReader>(reader: &mut T) -> Result<RTreeNodeStat> {
        return Ok(RTreeNodeStat {
            loss_chg: reader.read_f32_le()?,
            sum_hess: reader.read_f32_le()?,
            base_weight: reader.read_f32_le()?,
            leaf_child_cnt: reader.read_i32_le()?,
        });
    }
}

/// Regression tree
pub struct RegTree {
    param: Param,
    nodes: Vec<Node>,
    stats: Vec<RTreeNodeStat>,
}

impl RegTree {
    pub fn read_from<T: ModelReader>(reader: &mut T) -> Result<RegTree> {
        let param = Param::read_from(reader)?;
        let nodes: Result<Vec<Node>> = (0..param.num_nodes)
            .map(|_| Node::read_from(reader))
            .collect();
        let stats: Result<Vec<RTreeNodeStat>> = (0..param.num_nodes)
            .map(|_| RTreeNodeStat::read_from(reader))
            .collect();
        return Ok(RegTree {
            param,
            nodes: nodes?,
            stats: stats?,
        });
    }

    pub fn get_leaf_index<F: FVec>(&self, feat: &F, root_id: usize) -> usize {
        let mut pid = root_id;
        let mut node = self.nodes[pid];
        loop {
            match node.next(feat) {
                None => return pid,
                Some(new_pid) => {
                    pid = new_pid;
                    node = self.nodes[pid];
                }
            };
        }
    }

    pub fn get_leaf_value<F: FVec>(&self, feat: &F, root_id: usize) -> f32 {
        let leaf_node = self.nodes[self.get_leaf_index(feat, root_id)];
        return match leaf_node.leaf_or_split {
            LeafOrSplit::LeafValue(leaf_value) => leaf_value,
            LeafOrSplit::Split { .. } => {
                panic!("Broken tree - is not leaf node")
            }
        };
    }
}

impl Clone for RegTree {
    fn clone(&self) -> RegTree {
        return RegTree {
            param: self.param.clone(),
            nodes: self.nodes.clone(),
            stats: self.stats.clone(),
        };
    }
}
