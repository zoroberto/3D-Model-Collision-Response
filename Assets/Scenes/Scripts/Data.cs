﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Octree
{
    public class PairIndex
    {
        public List<int> index { get; set; }
        public int level { get; set; }

        public PairIndex(List<int> i)
        {
            index = i;
        }

        public PairIndex(List<int> i, int lv)
        {
            index = i;
            level = lv;
        }
    }

    public struct PairData
    {
        public int i1;
        public int i2;
    }

    public class VertexData
    {
        public Vector3[] pos;
    }

    public struct BoundData 
    {
        public Vector3 min;
        public Vector3 max;
    }

    public struct OctreeData
    {
        public Vector3 min;
        public Vector3 max;
        public Vector3 center;
        public Vector3 size;


        public Vector3 Minimum(Vector3 position, Vector3 size)
        {
            //Vector3 min = position - scale / 4; level 1
            //Vector3 min = position - scale / 8; level 2  
            Vector3 min = new Vector3(position.x - size.x, position.y - size.y, position.z - size.z);
            return min;
        }

        public Vector3 Maximum(Vector3 position, Vector3 scale)
        {

            //Vector3 max = position + scale / 4; level 1
            //Vector3 max = position + scale / 8; level 2
            Vector3 max = new Vector3(position.x + scale.x, position.y + scale.y, position.z + scale.z);
            return max;
        }

        public Vector3 Center(Vector3 Max, Vector3 Min)
        {
            return center = (Max + Min) / 2;
        }
        
        public Vector3 Size(Vector3 Max, Vector3 Min)
        {
            return size = Max - Min;
        }
    };

    public struct BoundingBox
    {
        public Vector3 min;
        public Vector3 max;
        public bool collide;
    }

    public class Intersection
    {

        public static bool AABB(OctreeData box1, OctreeData box2)
        {
            return
                box1.min.x <= box2.max.x &&
                box1.max.x >= box2.min.x &&
                box1.min.y <= box2.max.y &&
                box1.max.y >= box2.min.y &&
                box1.min.z <= box2.max.z &&
                box1.max.z >= box2.min.z;
        }
        
        public static bool AABB(Vector3 box1min, Vector3 box1max, Vector3 box2min, Vector3 box2max)
        {
            return
                box1min.x <= box2max.x &&
                box1max.x >= box2min.x &&
                box1min.y <= box2max.y &&
                box1max.y >= box2min.y &&
                box1min.z <= box2max.z &&
                box1max.z >= box2min.z;
        }
    }
}
