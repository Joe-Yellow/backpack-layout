//! Item bitsets. The exact solver keeps one u64 per state (at most 63 items, memory-bound); the
//! fast beam search can take up to 255 items with `Wide`.

use std::hash::Hash;
use std::ops::{BitAnd, BitOr, BitOrAssign, Not};

pub trait Bits:
    Copy
    + Default
    + Eq
    + Hash
    + Send
    + Sync
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitOrAssign
    + Not<Output = Self>
{
    /// Highest usable item index + 1.
    const CAPACITY: usize;
    fn bit(i: usize) -> Self;
    fn is_zero(self) -> bool;
    /// The lowest set bit alone (zero stays zero).
    fn lowest(self) -> Self;
    /// Index of the lowest set bit; only for non-zero sets.
    fn first(self) -> usize;

    /// Indices of the set bits, ascending.
    fn ones(self) -> impl Iterator<Item = usize> {
        let mut rest = self;
        std::iter::from_fn(move || {
            if rest.is_zero() {
                return None;
            }
            let i = rest.first();
            rest = rest & !Self::bit(i);
            Some(i)
        })
    }
}

impl Bits for u64 {
    const CAPACITY: usize = 63;
    fn bit(i: usize) -> Self {
        1 << i
    }
    fn is_zero(self) -> bool {
        self == 0
    }
    fn lowest(self) -> Self {
        self.isolate_lowest_one()
    }
    fn first(self) -> usize {
        self.trailing_zeros() as usize
    }
}

const WORDS: usize = 4;

#[derive(Clone, Copy, Default, PartialEq, Eq, Hash, Debug)]
pub struct Wide([u64; WORDS]);

impl Bits for Wide {
    const CAPACITY: usize = 64 * WORDS - 1;
    fn bit(i: usize) -> Self {
        let mut w = [0; WORDS];
        w[i / 64] = 1 << (i % 64);
        Wide(w)
    }
    fn is_zero(self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }
    fn lowest(self) -> Self {
        match self.0.iter().position(|&w| w != 0) {
            Some(k) => {
                let mut w = [0; WORDS];
                w[k] = self.0[k].lowest();
                Wide(w)
            }
            None => self,
        }
    }
    fn first(self) -> usize {
        let k = self.0.iter().position(|&w| w != 0).expect("empty set");
        64 * k + self.0[k].trailing_zeros() as usize
    }
}

impl BitAnd for Wide {
    type Output = Self;
    fn bitand(self, o: Self) -> Self {
        Wide(std::array::from_fn(|k| self.0[k] & o.0[k]))
    }
}

impl BitOr for Wide {
    type Output = Self;
    fn bitor(self, o: Self) -> Self {
        Wide(std::array::from_fn(|k| self.0[k] | o.0[k]))
    }
}

impl BitOrAssign for Wide {
    fn bitor_assign(&mut self, o: Self) {
        *self = *self | o;
    }
}

impl Not for Wide {
    type Output = Self;
    fn not(self) -> Self {
        Wide(self.0.map(|w| !w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_matches_bit_semantics_across_words() {
        let s = Wide::bit(3) | Wide::bit(64) | Wide::bit(200);
        assert_eq!(s.ones().collect::<Vec<_>>(), [3, 64, 200]);
        assert_eq!(s.lowest(), Wide::bit(3));
        assert_eq!((s & !Wide::bit(3)).lowest(), Wide::bit(64));
        assert_eq!((s & !Wide::bit(3)).first(), 64);
        assert!(Wide::default().is_zero() && Wide::default().lowest().is_zero());
        assert_eq!(0b1011_0000u64.ones().collect::<Vec<_>>(), [4, 5, 7]);
    }
}
