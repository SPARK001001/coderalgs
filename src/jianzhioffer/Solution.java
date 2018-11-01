package jianzhioffer;

import java.util.ArrayList;
import java.util.Stack;
import java.util.*;

public class Solution {
	/**
	 * 1⃣️ 二维数组中的查找 在一个二维数组中（每个一维数组的长度相同）， 每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。
	 * 请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
	 */
	// 每行用二分法
//	public boolean Find(int target, int[][] array) {
//
//		for (int i = 0; i < array.length; i++) {
//			int low = 0;
//			int high = array[i].length - 1;
//			while (low <= high) {
//				int mid = (low + high) / 2;
//				if (target > array[i][mid])
//					low = mid + 1;
//				else if (target < array[i][mid])
//					high = mid - 1;
//				else
//					return true;
//			}
//		}
//		return false;
//
//	}
	public boolean Find(int target, int[][] array) {
		if (array == null || array.length == 0)
			return false;
		int x = array.length - 1;// 二维数组左下角或右上角开始找
		int y = 0;
		while (x >= 0 && y < array[0].length) {
			if (array[x][y] == target)
				return true;
			if (array[x][y] > target)
				x--;
			else
				y++;
		}
		return false;
	}

	/**
	 * 2⃣⃣️ 替换空格 请实现一个函数，将一个字符串中的每个空格替换成“%20”。 例如，当字符串为We Are
	 * Happy.则经过替换之后的字符串为We%20Are%20Happy。
	 */
//	public String replaceSpace(StringBuffer str) {
//        String s = str.toString();
//        return s.replaceAll(" ","%20");
//    }

	public String replaceSpace(StringBuffer str) {
		String s = str.toString();
		char[] ch = s.toCharArray();
		int blank_count = 0;
		for (char c : ch) {
			if (c == ' ')// char类型空用单引号
				blank_count++;
		}
		int newLen = ch.length + 2 * blank_count;
		char[] newArray = new char[newLen];
		for (int i = ch.length - 1; i >= 0; i--) {
			if (ch[i] != ' ')
				newArray[--newLen] = ch[i];
			else {
				newArray[--newLen] = '0';
				newArray[--newLen] = '2';
				newArray[--newLen] = '%';
			}
		}
		return new String(newArray);// 字节数组变成字符串方法
	}

	/**
	 * 3⃣️ 从头到尾打印链表 输入一个链表，按链表值从尾到头的顺序返回一个ArrayList。
	 * 
	 * @author pjr
	 *
	 */

	class ListNode {
		int val;
		ListNode next = null;

		ListNode(int val) {
			this.val = val;
		}
	}

	// 23ms 9276k

	public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {

		ArrayList<Integer> list = new ArrayList<Integer>();
		Stack<Integer> stack = new Stack<Integer>();// 要加范型不然下边add不让
		while (listNode != null) {
			stack.push(listNode.val);
			listNode = listNode.next;
		}
		while (!stack.empty()) {
			list.add(stack.pop());// 不是append方法
		}
		return list;
	}

	/**
	 * 4.重建二叉树️⃣ 题目描述 输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。 假设输入的前序遍历和中序遍历的结果中都不含重复的数字。
	 * 例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
	 * 
	 */
	public class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			val = x;
		}
	}

	public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
		TreeNode root = reConstructBinaryTree(pre, 0, pre.length - 1, in, 0, in.length - 1);
		return root;
	}

	// 前序遍历{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}
	private TreeNode reConstructBinaryTree(int[] pre, int startPre, int endPre, int[] in, int startIn, int endIn) {

		if (startPre > endPre || startIn > endIn)
			return null;
		TreeNode root = new TreeNode(pre[startPre]);

		for (int i = startIn; i <= endIn; i++)
			if (in[i] == pre[startPre]) {
				root.left = reConstructBinaryTree(pre, startPre + 1, startPre + i - startIn, in, startIn, i - 1);
				root.right = reConstructBinaryTree(pre, i - startIn + startPre + 1, endPre, in, i + 1, endIn);
				break;
			}

		return root;
	}

	/**
	 * 5.两个栈实现队列 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。 java语言中
	 * 针对数组提供了length属性来获取数组的长度
	 * 
	 * 针对字符串提供了length()方法来获取字符串的长度
	 * 
	 * 针对泛型集合类提供了size()方法来获取元素的个数
	 */

	Stack<Integer> stack1 = new Stack<Integer>();
	Stack<Integer> stack2 = new Stack<Integer>();

	public void push(int node) {
		while (!stack2.isEmpty()) {// 注意没有size()方法
			stack1.push(stack2.pop());
		}
		stack1.push(node);
	}

	public int pop() {
		while (!stack1.isEmpty()) {
			stack2.push(stack1.pop());
		}
		return stack2.pop();
	}

	/**
	 * 6️⃣旋转数组的最小数字
	 * 
	 * 把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转，输出旋转数组的最小元素。
	 * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。 NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。
	 */
	// 335 23732
	public int minNumberInRotateArray(int[] array) {
//        if(array.length==0)
//            return 0;
//        for(int i=0;i<array.length;i++){
//            if(array[i+1]<array[i])
//                return array[i+1];
//        }
//        return array[0];
//    }

		if (array.length == 0)
			return 0;
		int low = 0;
		int high = array.length - 1;

		while (low < high) {
			int mid = low + (high - low) / 2;
			if (array[mid] > array[high])
				low = mid + 1;
			else if (array[mid] == array[high])

				high = high - 1;// 出现这种情况的array类似 [1,0,1,1,1] 或者[1,1,1,0,1]，此时最小数字不好判断在mid左边还是右边,这时只好一个一个试 ，
			else
				high = mid;
		}
		return array[low];
	}

	/**
	 * 7️⃣ 斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
	 * 
	 * @param n
	 * @return
	 */
	// 循环 动态规划
	/*
	 * public int Fibonacci(int n) {
	 * 
	 * 
	 * if(n<=1) return n; int[] record = new int[n+1]; record[0]=0; record[1]=1;
	 * for(int i=2;i<=n;i++){ record[i]=record[i-1]+record[i-2]; } return record[n];
	 * }
	 */

	// 尾递归
	public int Fibonacci(int n) {

		return Fibonacci(n, 0, 1);
	}

	public int Fibonacci(int n, int i, int j) {
		if (n == 0)
			return i;
		if (n == 1)
			return j;
		return Fibonacci(n - 1, j, i + j);
	}

	// 8️⃣青蛙跳台阶 一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。
	public int JumpFloor(int target) {
		if (target <= 2)
			return target;
		// return JumpFloor(target-1)+JumpFloor(target-2);
		int first = 1;
		int second = 2;
		int third = 0;
		for (int i = 3; i <= target; i++) {
			third = first + second;
			first = second;
			second = third;
		}
		return second;
	}

	// 9一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。️⃣ int a=1; return
	// a<<(number-1);
	public int JumpFloorII(int target) {
		if (target <= 1)
			return target;
		return 2 * JumpFloorII(target - 1);
	}

	// 10.矩形覆盖
	// 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

	/*
	 * public int RectCover(int target) {//595ms 9220k 递归
	 * 
	 * if(target<=0) return 0; if(target==1) return 1; if(target==2) return 2; else
	 * return RectCover(target-1)+RectCover(target-2); }
	 */
	public int RectCover(int target) { // 尾递归 14ms 9420k
		return RectCover(target, 1, 2);
	}

	private int RectCover(int target, int i, int j) {
		if (target <= 0)
			return 0;
		if (target == 1)
			return i;
		if (target == 2)
			return j;
		else
			return RectCover(target - 1, j, i + j);
	}

	// 11.二进制中1的个数
	// 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。巧妙⭐⭐
	// 若考虑n右移，负数的时候>>左边会补1，陷入死循环可以用无符号右移>>>; 也可以考虑设置flag为1，每次左移一位与n相与️
	public int NumberOf1(int n) {

		int count = 0;

		while (n != 0) {
			++count;
			n = n & (n - 1);
		}
		return count;
	}

	// 12.数值的整数次方 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。
//	public double Power(double base, int exponent) { //使用累乘的方法，时间复杂度O(n) 32ms 10560k
//        double count = 1;
//        int n = Math.abs(exponent);
//        while(n!=0){
//            --n;
//            count = count*base;
//        }
//        if(exponent<0)
//            count = 1/count;
//        return count;
//  }
	public double Power(double base, int exponent) { // 使用递归O(logN)
		double count = 1.0;
		int n = Math.abs(exponent);
		if (n == 0)
			return 1;
		if (n == 1)
			return base;
		count = Power(base, n >> 1);
		count *= count;
		if ((n & 1) == 1)
			count *= base;
		if (exponent < 0)
			count = 1 / count;
		return count;
	}
	 /*13.调整数组顺序使奇数位于偶数前面
	 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，
	 使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，
	 并保证奇数和奇数，偶数和偶数之间的相对位置不变。
	 */
	 public void reOrderArray(int [] array) {
	        
	        int len = array.length;
	        int[] newArr = new int[len];
	        int point = 0;
	        for(int i=0;i<array.length;i++){//i<len不可以，下面len会变的
	            if(array[i]%2==1){
	                newArr[point++]=array[i]; //newArr[i++]=array[i];i不是newArr中的
	            }else{
	                newArr[--len]=array[i];
	            }
	        }
	        for(int j=0;j<point;j++){ //j<=point不要加=,point++，最后会自增1的
	            array[j] = newArr[j];
	        }
	        len = array.length; //重置len;
	        for(int j=point;j<array.length;j++){
	            array[j] = newArr[--len];
	        }
	    }
		
	/*public void reOrderArray(int [] array) {
	        //插入排序的思想，若是奇数插入到第一个偶数的前面
	        for(int i=0;i<array.length;i++){
	            int temp = array[i];
	            int j = i-1;
	            if(array[i]%2==1){
	                while(j>=0&&array[j]%2==0){
	                    array[j+1] = array[j];
	                    j--;
	                }
	            }
	            array[j+1] = temp;
	        }
	    }
	    */
		
	//14.链表中倒数第k个结点;输入一个链表，输出该链表中倒数第k个结点。
	 public ListNode FindKthToTail(ListNode head,int k) {
	        if(head==null||k<=0)//<=0
	            return null;
	        ListNode node1 = head;
	        ListNode node2 = head;
	        for(int i=1;i<k;i++){ 
	            if(node1.next!=null){
	                node1 = node1.next;
	            }else{
	                return null;//在此处判断k大于链表长度返回null
	            }
	        }
	        while(node1.next!=null){
	            node1 = node1.next;
	            node2 = node2.next;
	        }
	        return node2;
	    }
		
	//15.翻转链表输入一个链表，反转链表后，输出新链表的表头。☆☆☆
	public ListNode ReverseList(ListNode head) {
	        if(head==null)
	            return null;
	        ListNode pre = null;//定义两个结点用来保存翻转之后的链表。
	        ListNode next = null;
	        while(head!=null){
	            next = head.next;
	            head.next = pre;
	            pre = head;
	            head = next;
	        }
	        return pre;
	    }
		
	//16.合并两个排序的链表  输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。
	 public ListNode Merge(ListNode list1,ListNode list2) {
	        if(list1==null && list2==null)
	            return null;
	        
	        ListNode head = new ListNode(-1);//自己在如何返回头结点处碰到困难，不能直接返回head
	        head.next = null;
	        ListNode root = head;
	        
	        while(list1!=null && list2!=null){
	            if(list1.val>list2.val && list2!=null){
	                head.next = list2;
	                head = list2;
	                list2 = list2.next;
	            }else if(list1.val<=list2.val && list1!=null){
	                head.next = list1;
	                head = list1;
	                list1 = list1.next;
	            }
	        }
	        if(list1 == null)
	            head.next = list2;
	        else if(list2 == null)
	            head.next = list1;
	        return root.next;
	    }
	//17.输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

	public static boolean HasSubtree(TreeNode root1, TreeNode root2) {
	        boolean result = false;
	        //当Tree1和Tree2都不为零的时候，才进行比较。否则直接返回false
	        if (root2 != null && root1 != null) {
	            //如果找到了对应Tree2的根节点的点
	            if(root1.val == root2.val){
	                //以这个根节点为为起点判断是否包含Tree2
	                result = doesTree1HaveTree2(root1,root2);
	            }
	            //如果找不到，那么就再去root的左儿子当作起点，去判断时候包含Tree2
	            if (!result) {
	                result = HasSubtree(root1.left,root2);
	            }
	             
	            //如果还找不到，那么就再去root的右儿子当作起点，去判断时候包含Tree2
	            if (!result) {
	                result = HasSubtree(root1.right,root2);
	               }
	            }
	            //返回结果
	        return result;
	    }
	 
	    public static boolean doesTree1HaveTree2(TreeNode node1, TreeNode node2) {
	        //如果Tree2已经遍历完了都能对应的上，返回true
	        if (node2 == null) {
	            return true;
	        }
	        //如果Tree2还没有遍历完，Tree1却遍历完了。返回false
	        if (node1 == null) {
	            return false;
	        }
	        //如果其中有一个点没有对应上，返回false
	        if (node1.val != node2.val) {  
	                return false;
	        }
	         
	        //如果根节点对应的上，那么就分别去子节点里面匹配
	        return doesTree1HaveTree2(node1.left,node2.left) && doesTree1HaveTree2(node1.right,node2.right);
	    }
		
	//18、二叉树的镜像
	//递归版本；
	public void Mirror(TreeNode root) {
	        if(root==null)
	            return ;
	        TreeNode temp = root.left;
	        root.left = root.right;
	        root.right = temp;
	        if(root.left!=null) Mirror(root.left);
	        if(root.right!=null) Mirror(root.right);
	    }
		
	//非递归版本，用一个栈
	/*public void Mirror(TreeNode root) {
	        if(root==null)
	            return ;
	        
	        Stack<TreeNode> stack = new Stack<>();
	        stack.push(root);
	        TreeNode tree = null;
			
	        while(!stack.empty()){
	            tree = stack.pop();
	            if(tree.left!=null || tree.right!=null){//交换左右结点
	                TreeNode temp = tree.left;
	                tree.left = tree.right;
	                tree.right = temp;
	            }
	            if(tree.left!=null)
	                stack.push(tree.left);
	            if(tree.right!=null)
	                stack.push(tree.right);
	        }
	    }
	    */
		
	//19.顺时针打印矩阵。输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字


	public ArrayList<Integer> printMatrix(int [][] array) {
	        ArrayList<Integer> result = new ArrayList<Integer> ();
	        if(array.length==0) return result;
	        int n = array.length,m = array[0].length;
	        if(m==0) return result;
	        int layers = (Math.min(n,m)-1)/2+1;//这个是层数
	        for(int i=0;i<layers;i++){
	            for(int k = i;k<m-i;k++) result.add(array[i][k]);//左至右
	            for(int j=i+1;j<n-i;j++) result.add(array[j][m-i-1]);//右上至右下
	            for(int k=m-i-2;(k>=i)&&(n-i-1!=i);k--) result.add(array[n-i-1][k]);//右至左
	            for(int j=n-i-2;(j>i)&&(m-i-1!=i);j--) result.add(array[j][i]);//左下至左上
	        }
	        return result;       
	    }
		
	//20.包含min函数的栈  定义栈的数据结构，请在该类型中实现一个能够得到栈中所含最小元素的min函数（时间复杂度应为O（1））。
	/*import java.util.Stack;

	public class Solution {
	    Stack<Integer> stack1 = new Stack<>(); 
	    Stack<Integer> stack2 = new Stack<>(); 
	    int min = Integer.MAX_VALUE;
	    public void push(int node) {
	        if(node<min){
	            min = node;
	            stack1.push(min);
	        }
	        stack2.push(node);
	    }
	    
	    public void pop() {
	        if(stack2.peek()==stack1.peek()){
	            stack1.pop();
	        }
	        stack2.pop();
	    }
	    
	    public int top() {
	        return stack2.peek();
	    }
	    
	    public int min() {
	        return stack1.peek();
	    }
	}
	*/

	

//21.栈的压人弹出序列 
//输入两个整数序列，第一个序列表示栈的压入顺序，请判断第二个序列是否可能为该栈的弹出顺序。假设压入栈的所有数字均不相等。
//例如序列1,2,3,4,5是某栈的压入顺序，序列4,5,3,2,1是该压栈序列对应的一个弹出序列，但4,3,5,1,2就不可能是该压栈序列的弹出序列。（注意：这两个序列的长度是相等的）

	public boolean IsPopOrder(int[] pushA, int[] popA) {
		if (pushA.length == 0 || popA.length == 0)
			return false;
		Stack<Integer> s = new Stack<Integer>();
		// 用于标识弹出序列的位置
		int popIndex = 0;
		for (int i = 0; i < pushA.length; i++) {
			s.push(pushA[i]);
			// 如果栈不为空，且栈顶元素等于弹出序列
			while (!s.empty() && s.peek() == popA[popIndex]) {
				// 出栈
				s.pop();
				// 弹出序列向后一位
				popIndex++;
			}
		}
		return s.empty();
	}

//22.从上往下打印二叉树。从上往下打印出二叉树的每个节点，同层节点从左至右打印。
//思路是用arraylist模拟一个队列来存储相应的TreeNode 每层的node

	public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
		ArrayList<Integer> list = new ArrayList<>();
		ArrayList<TreeNode> queue = new ArrayList<>();
		if (root == null) {
			return list;//空时返回空list
		}
		queue.add(root);
		while (queue.size() != 0) {//queue.isEmpty()  stack.empty()
			TreeNode temp = queue.remove(0);
			if (temp.left != null) {
				queue.add(temp.left);
			}
			if (temp.right != null) {
				queue.add(temp.right);
			}
			list.add(temp.val);
		}
		return list;
	}
//23.二叉搜索树的后序遍历序列 输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
	public boolean VerifySquenceOfBST(int [] sequence) {
        int len = sequence.length;
        if(len==0) return false;
        return VerifySquenceOfBST(sequence,0,len-1);
        
        }
    public boolean VerifySquenceOfBST(int[] arr,int start, int end){
        if(start>=end) return true; //start==end对应的是叶子结点，start>end对应的是空树，这两种情况都是合法的二叉搜索树
        while(arr[start]<arr[end]){
            start++;
        }
        int temp = start; 
        for(int i=temp;i<end-1;i++){
            if(arr[i]<arr[end]) return false;
        }
        return VerifySquenceOfBST(arr,0,temp-1)&&VerifySquenceOfBST(arr,temp,end-1);
    }
    //非递归版本
    	//非递归也是一个基于递归的思想：
    	//左子树一定比右子树小，因此去掉根后，数字分为left，right两部分，right部分的
    	//最后一个数字是右子树的根他也比左子树所有值大，因此我们可以每次只看有子树是否符合条件
    	//即可，即使到达了左子树左子树也可以看出由左右子树组成的树还想右子树那样处理
    	 
    	//对于左子树回到了原问题，对于右子树，左子树的所有值都比右子树的根小可以暂时把他看出右子树的左子树
    	//只需看看右子树的右子树是否符合要求即可
/*    
 	public boolean VerifySquenceOfBST(int [] sequence) {
        
        int size = sequence.length;
        if(0==size) return false;
 
        int i = 0;
        while(--size!=0)
        {
            while(sequence[i]<sequence[size]){i++;}
            while(sequence[i]>sequence[size]){i++;}
 
            if(i<size)return false;
            i=0;
        }
        return true;
    }
*/
    
//24.二叉树中和为某一值的路径
//输入一颗二叉树的跟节点和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。(注意: 在返回值的list中，数组长度大的数组靠前)

    	 private ArrayList<ArrayList<Integer>> listAll = new ArrayList<ArrayList<Integer>>();
    	    private ArrayList<Integer> list = new ArrayList<Integer>();
    	    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
    	        if(root == null) return listAll;
    	        list.add(root.val);
    	        target -= root.val;
    	        if(target == 0 && root.left == null && root.right == null)
    	            listAll.add(new ArrayList<Integer>(list));
    	        FindPath(root.left, target);
    	        FindPath(root.right, target);
    	        list.remove(list.size()-1);//返回上一层
    	        return listAll;
    	    }
    	    
    	   /* 链接：https://www.nowcoder.com/questionTerminal/b736e784e3e34731af99065031301bca
    	    	来源：牛客网

    	    	public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {
    	    	        ArrayList<ArrayList<Integer>> paths=new ArrayList<ArrayList<Integer>>();
    	    	        if(root==null)return paths;
    	    	        find(paths,new ArrayList<Integer>(),root,target);
    	    	        return paths;
    	    	    }
    	    	    public void find(ArrayList<ArrayList<Integer>> paths,ArrayList<Integer> path,TreeNode root,int target){
    	    	        path.add(root.val);
    	    	        if(root.left==null&&root.right==null){
    	    	            if(target==root.val){
    	    	                paths.add(path);
    	    	            }
    	    	            return;
    	    	        }
    	    	        ArrayList<Integer> path2=new ArrayList<>();
    	    	        path2.addAll(path);
    	    	        if(root.left!=null)find(paths,path,root.left,target-root.val);//没有返回上一层是因为target在下一层才更新。
    	    	        if(root.right!=null)find(paths,path2,root.right,target-root.val);
    	    	    } */
//25.⭐⭐⭐复杂链表的复制；
//输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）
    	    public class RandomListNode {
    	        int label;
    	        RandomListNode next = null;
    	        RandomListNode random = null;

    	        RandomListNode(int label) {
    	            this.label = label;
    	        }
    	    }
    	    public RandomListNode Clone(RandomListNode pHead)
    	    {
    	       
    	        HashMap<RandomListNode,RandomListNode> map = new HashMap<RandomListNode,RandomListNode>();
    	        RandomListNode p = pHead;
    	        RandomListNode q = new RandomListNode(-1);
    	        while(p!=null){
    	            RandomListNode t = new RandomListNode(p.label);
    	            map.put(p, t);
    	            p = p.next;
    	            q.next = t;//注意赋值
    	            q = t;
    	        }
    	      

    	        p = pHead;
    	        while (p != null) {
    	            map.get(p).random = map.get(p.random);//注意用
    	            p = p.next;
    	        }
    	        return map.get(pHead);
    	    }
    	    
    	    /* 优化算法，不用o(n)的hashmap
    	    	
    	    	*解题思路：
    	    	*1、遍历链表，复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；
    	    	*2、重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;
    	    	*3、拆分链表，将链表拆分为原链表和复制后的链表
    	    	
    	    	public class Solution {
    	    	    public RandomListNode Clone(RandomListNode pHead) {
    	    	        if(pHead == null) {
    	    	            return null;
    	    	        }
    	    	         
    	    	        RandomListNode currentNode = pHead;
    	    	        //1、复制每个结点，如复制结点A得到A1，将结点A1插到结点A后面；
    	    	        while(currentNode != null){
    	    	            RandomListNode cloneNode = new RandomListNode(currentNode.label);
    	    	            RandomListNode nextNode = currentNode.next;
    	    	            currentNode.next = cloneNode;
    	    	            cloneNode.next = nextNode;
    	    	            currentNode = nextNode;
    	    	        }
    	    	         
    	    	        currentNode = pHead;
    	    	        //2、重新遍历链表，复制老结点的随机指针给新结点，如A1.random = A.random.next;
    	    	        while(currentNode != null) {
    	    	            currentNode.next.random = currentNode.random==null?null:currentNode.random.next;
    	    	            currentNode = currentNode.next.next;
    	    	        }
    	    	         
    	    	        //3、拆分链表，将链表拆分为原链表和复制后的链表
    	    	        currentNode = pHead;
    	    	        RandomListNode pCloneHead = pHead.next;
    	    	        while(currentNode != null) {
    	    	            RandomListNode cloneNode = currentNode.next;
    	    	            currentNode.next = cloneNode.next;
    	    	            cloneNode.next = cloneNode.next==null?null:cloneNode.next.next;
    	    	            currentNode = currentNode.next;
    	    	        }
    	    	         
    	    	        return pCloneHead;
    	    	    }
    	    	}
    	    	*/
    	    
  //⭐⭐⭐⭐26.二叉搜索树与双向链表	  输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。
    	   

    	    	//思路与方法二中的递归版一致，仅对第2点中的定位作了修改，新增一个全局变量记录左子树的最后一个节点。
    	    	    // 记录子树链表的最后一个节点，终结点只可能为只含左子树的非叶节点与叶节点
    	    	    protected TreeNode leftLast = null;
    	    	    public TreeNode Convert(TreeNode root) {
    	    	        if(root==null)
    	    	            return null;
    	    	        if(root.left==null&&root.right==null){
    	    	            leftLast = root;// 最后的一个节点可能为最右侧的叶节点
    	    	            return root;
    	    	        }
    	    	        // 1.将左子树构造成双链表，并返回链表头节点
    	    	        TreeNode left = Convert(root.left);
    	    	        // 3.如果左子树链表不为空的话，将当前root追加到左子树链表
    	    	        if(left!=null){
    	    	            leftLast.right = root;
    	    	            root.left = leftLast;
    	    	        }
    	    	        leftLast = root;// 当根节点只含左子树时，则该根节点为最后一个节点
    	    	        // 4.将右子树构造成双链表，并返回链表头节点
    	    	        TreeNode right = Convert(root.right);
    	    	        // 5.如果右子树链表不为空的话，将该链表追加到root节点之后
    	    	        if(right!=null){
    	    	            right.left = root;
    	    	            root.right = right;
    	    	        }
    	    	        return left!=null?left:root;       
    	    	    }
 //⭐⭐⭐️27.字符串的排列  
    	    	    //输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。
    	    	  

    	    	    	public ArrayList<String> Permutation(String str) {
    	    	    	        ArrayList<String> re = new ArrayList<String>();
    	    	    	        if (str == null || str.length() == 0) {
    	    	    	            return re;
    	    	    	        }
    	    	    	        HashSet<String> set = new HashSet<String>();
    	    	    	        fun(set, str.toCharArray(), 0);
    	    	    	        re.addAll(set);
    	    	    	        Collections.sort(re);
    	    	    	        return re;
    	    	    	    }
    	    	    	    void fun(HashSet<String> re, char[] str, int k) {
    	    	    	        if (k == str.length) {
    	    	    	            re.add(new String(str));
    	    	    	            return;
    	    	    	        }
    	    	    	        for (int i = k; i < str.length; i++) {
    	    	    	            swap(str, i, k);
    	    	    	            fun(re, str, k + 1);
    	    	    	            swap(str, i, k); 

    	    	    	            /*举例来说“abca”，为什么使用了两次swap函数
    	    	    	                            交换时是a与b交换，遍历；
    	    	    	                            交换时是a与c交换，遍历；（使用一次swap时，是b与c交换）
    	    	    	                            交换时是a与a不交换；
    	    	    	                            */
    	    	    	        }
    	    	    	    }
    	    	    	    void swap(char[] str, int i, int j) {
    	    	    	        if (i != j) {
    	    	    	            char t = str[i];
    	    	    	            str[i] = str[j];
    	    	    	            str[j] = t;
    	    	    	        }
    	    	    	    }
  //28.数组中出现次数超过一半的数字数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。
    	    	    	    public int MoreThanHalfNum_Solution(int [] array) {
    	    	    	        
//    	    	    	    	HashMap<Integer,Integer> map = new HashMap<>();
//    	    	    	    	Set keys = map.keySet();不能这样，map还没初始化，keys里面啥都没；
//    	    	    	    	for(int i=0;i<array.length;i++){
//    	    	    	    		if(keys.containsKey(array[i])){
    	    	    	        HashMap<Integer,Integer> map = new HashMap<>();
    	    	    	        for(int i=0;i<array.length;i++){
    	    	    	            if(map.containsKey(array[i])){

    	    	    	                map.put(array[i],map.get(array[i])+1);
    	    	    	            }else{
    	    	    	                map.put(array[i],1);
    	    	    	            }
    	    	    	        }
    	    	    	        Set<Map.Entry<Integer,Integer>> entrySet = map.entrySet();
    	    	    	        Iterator<Map.Entry<Integer,Integer>> it = entrySet.iterator();
    	    	    	        while(it.hasNext()){
    	    	    	            Map.Entry<Integer,Integer> entry = it.next();
    	    	    	            if(entry.getValue()>array.length/2){
    	    	    	                return entry.getKey();
    	    	    	            }
    	    	    	        }
    	    	    	        return 0;
    	    	    	    }
    	    	    	   /* 链接：https://www.nowcoder.com/questionTerminal/e8a1b01a2df14cb2b228b30ee6a92163
    	    	    	    	来源：牛客网

    	    	    	    	public int MoreThanHalfNum_Solution(int [] array) {
    	    	    	    	        HashMap<Integer,Integer> map = new HashMap<Integer,Integer>();
    	    	    	    	         
    	    	    	    	        for(int i=0;i<array.length;i++){
    	    	    	    	             
    	    	    	    	            if(!map.containsKey(array[i])){
    	    	    	    	               map.put(array[i],1);
    	    	    	    	            }else{
    	    	    	    	                int count = map.get(array[i]);
    	    	    	    	                map.put(array[i],++count);
    	    	    	    	            }
    	    	    	    	        }
    	    	    	    	        Iterator iter = map.entrySet().iterator();
    	    	    	    	        while(iter.hasNext()){
    	    	    	    	            Map.Entry entry = (Map.Entry)iter.next();
    	    	    	    	            Integer key =(Integer)entry.getKey();
    	    	    	    	            Integer val = (Integer)entry.getValue();
    	    	    	    	            if(val>array.length/2){
    	    	    	    	                return key;
    	    	    	    	            }
    	    	    	    	        }
    	    	    	    	        return 0; */
    	    	    	    
  //29.最小的k个数；输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。
    	    	    	 

	//用最大堆保存这k个数，每次只和堆顶比，如果比堆顶小，删除堆顶，新数入堆。
	
	//import java.util.PriorityQueue;
	//import java.util.Comparator;

	public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		int length = input.length;
		if (k > length || k == 0) {
			return result;
		}
		PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k, new Comparator<Integer>() {

			@Override
			public int compare(Integer o1, Integer o2) {
				return o2.compareTo(o1);
			}
		});
		for (int i = 0; i < length; i++) {
			if (maxHeap.size() != k) {
				maxHeap.offer(input[i]);
			} else if (maxHeap.peek() > input[i]) {
				Integer temp = maxHeap.poll();
				temp = null;
				maxHeap.offer(input[i]);
			}
		}
		for (Integer integer : maxHeap) {
			result.add(integer);
		}
		return result;
	}
//30.连续子数组的最大和  例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和
	public int FindGreatestSumOfSubArray(int[] array) {
        if(array==null || array.length==0) return 0;
        int count = 0;//不要写成array[0]
        int max = Integer.MIN_VALUE;
        for(int i=0;i<array.length;i++){
            if(count<=0){
                count=array[i];
            }else{
                count+=array[i];
            }
            if(count>max){//这里每次循环都记录
                max = count;
            }
        }
        return max;
    }
	
//31.整数中1出现的次数 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次
	/*
	 * 

		像类似这样的问题，我们可以通过归纳总结来获取相关的东西。
		首先可以先分类：
		我们知道在个位数上，1会每隔10出现一次，例如1、11、21等等，我们发现以10为一个阶梯的话，每一个完整的阶梯里面都有一个1，例如数字22，按照10为间隔来分三个阶梯，在完整阶梯0-9，10-19之中都有一个1，但是19之后有一个不完整的阶梯，我们需要去判断这个阶梯中会不会出现1，易推断知，如果最后这个露出来的部分小于1，则不可能出现1（这个归纳换做其它数字也成立）。
		我们可以归纳个位上1出现的个数为：n/10 * 1+(n%10!=0 ? 1 : 0)。
		现在说十位数，十位数上出现1的情况应该是10-19，依然沿用分析个位数时候的阶梯理论，我们知道10-19这组数，每隔100出现一次，这次我们的阶梯是100，例如数字317，分析有阶梯0-99，100-199，200-299三段完整阶梯，每一段阶梯里面都会出现10次1（从10-19），最后分析露出来的那段不完整的阶梯。我们考虑如果露出来的数大于19，那么直接算10个1就行了，因为10-19肯定会出现；如果小于10，那么肯定不会出现十位数的1；如果在10-19之间的，我们计算结果应该是k - 10 + 1。例如我们分析300-317，17个数字，1出现的个数应该是17-10+1=8个。那么现在可以归纳：十位上1出现的个数为：
		·设k= n % 100，即为不完整阶梯段的数字
		·归纳式为：(n / 100) * 10 + (if(k > 19) 10 else if(k < 10) 0 else k - 10 + 1)
		现在说百位1，我们知道在百位，100-199都会出现百位1，一共出现100次，阶梯间隔为1000，100-199这组数，每隔1000就会出现一次。这次假设我们的数为2139。跟上述思想一致，先算阶梯数 * 完整阶梯中1在百位出现的个数，即n/1000 * 100得到前两个阶梯中1的个数，那么再算漏出来的部分139，沿用上述思想，不完整阶梯数k<100则得到0个百位1，k>199，得到100个百位1，100<=k<=199则得到k - 100 + 1个百位1。那么继续归纳：百位上出现1的个数：
		·设k = n % 1000
		·归纳式为：(n / 1000) * 100 + (if(k >199) 10 else if(k < 100) 0 else k - 100 + 1)
		后面的依次类推....
		那么我们把个位数上算1的个数的式子也纳入归纳式中
		·k = n % 10
		·个位数上1的个数为：n / 10 * 1 + (if(k > 1) 1 else if(k < 1) 0 else k - 1 + 1)
		完美！归纳式看起来已经很规整了。
		来一个更抽象的归纳，设i为计算1所在的位数，i=1表示计算个位数的1的个数，10表示计算十位数的1的个数等等。
		·k = n % (i * 10)
		·count(i) = (n / (i * 10)) * i + (if(k > i * 2 - 1) i else if(k < i) 0 else k - i + 1)
		好了，这样从10到10的n次方的归纳就完成了。
		sum1 = sum(count(i))，i = Math.pow(10, j), 0<=j<=log10(n)
		
		但是有一个地方值得我们注意的，就是代码的简洁性来看，有多个ifelse不太好，能不能进一步简化呢？
		我们可以把后半段简化成这样，我们不去计算i * 2 - 1了，我们只需保证k - i + 1在[0, i]区间内就行了，最后后半段可以携程这样
		min(max((n mod (i*10))−i+1,0),i)
	 */

		 public int NumberOf1Between1AndN_Solution(int n) {
		        if(n <= 0)
		            return 0;
		        int count = 0;
		        for(long i = 1; i <= n; i *= 10){
		            long diviver = i * 10;
		            count += (n / diviver) * i + Math.min(Math.max(n % diviver - i + 1, 0), i);
		        }
		        return count;
		    } 
//32.把数组排成最小的数 
		 //输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。
		 /*
					  * 链接：https://www.nowcoder.com/questionTerminal/8fecd3f8ba334add803bf2a06af1b993
			来源：牛客网
			
			 * 解题思路：
			 * 先将整型数组转换成String数组，然后将String数组排序，最后将排好序的字符串数组拼接出来。关键就是制定排序规则。
			 * 排序规则如下：
			 * 若ab > ba 则 a > b，
			 * 若ab < ba 则 a < b，
			 * 若ab = ba 则 a = b；
			 * 解释说明：
			 * 比如 "3" < "31"但是 "331" > "313"，所以要将二者拼接起来进行比较
			public String PrintMinNumber(int [] numbers) {
			        if(numbers == null || numbers.length == 0) return "";
			        int len = numbers.length;
			        String[] str = new String[len];
			        StringBuilder sb = new StringBuilder();
			        for(int i = 0; i < len; i++){
			            str[i] = String.valueOf(numbers[i]);
			        }
			        Arrays.sort(str,new Comparator<String>(){
			            @Override
			            public int compare(String s1, String s2) {
			                String c1 = s1 + s2;
			                String c2 = s2 + s1;
			                return c1.compareTo(c2);
			            }
			        });
			        for(int i = 0; i < len; i++){
			            sb.append(str[i]);
			        }
			        return sb.toString();
			    }
			    
			    import java.util.ArrayList;
				import java.util.Collections;
				import java.util.Comparator;
				import java.util.Arrays;

		  */
		 public String PrintMinNumber(int [] numbers) {
		        if(numbers==null) return null;
		        StringBuilder sb = new StringBuilder();
		        ArrayList<Integer> list = new ArrayList<>();
		        
		        for(int i=0;i<numbers.length;i++){
		            list.add(numbers[i]);
		        }
		        Collections.sort(list,new Comparator<Integer>(){//Collections.sort(list...)  Arrays.sort(arr)
		            @Override
		            public int compare(Integer o1,Integer o2){
		                String s1 = o1+""+o2;
		                String s2 = o2+""+o1;
		                return s1.compareTo(s2);
		            }
		        });
		        for(Integer i:list){
		            sb.append(String.valueOf(i));
		        }
		        return sb.toString();
		    }
//33.丑数   把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。
		 
		 /*
		  * 链接：https://www.nowcoder.com/questionTerminal/6aa9e04fc3794f68acf8778237ba065b
				来源：牛客网
				
				通俗易懂的解释：
				首先从丑数的定义我们知道，一个丑数的因子只有2,3,5，那么丑数p = 2 ^ x * 3 ^ y * 5 ^ z，换句话说一个丑数一定由另一个丑数乘以2或者乘以3或者乘以5得到，那么我们从1开始乘以2,3,5，就得到2,3,5三个丑数，在从这三个丑数出发乘以2,3,5就得到4，6,10,6，9,15,10,15,25九个丑数，我们发现这种方法会得到重复的丑数，而且我们题目要求第N个丑数，这样的方法得到的丑数也是无序的。那么我们可以维护三个队列：
				（1）丑数数组： 1
				乘以2的队列：2
				乘以3的队列：3
				乘以5的队列：5
				选择三个队列头最小的数2加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
				（2）丑数数组：1,2
				乘以2的队列：4
				乘以3的队列：3，6
				乘以5的队列：5，10
				选择三个队列头最小的数3加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
				（3）丑数数组：1,2,3
				乘以2的队列：4,6
				乘以3的队列：6,9
				乘以5的队列：5,10,15
				选择三个队列头里最小的数4加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
				（4）丑数数组：1,2,3,4
				乘以2的队列：6，8
				乘以3的队列：6,9,12
				乘以5的队列：5,10,15,20
				选择三个队列头里最小的数5加入丑数数组，同时将该最小的数乘以2,3,5放入三个队列；
				（5）丑数数组：1,2,3,4,5
				乘以2的队列：6,8,10，
				乘以3的队列：6,9,12,15
				乘以5的队列：10,15,20,25
				选择三个队列头里最小的数6加入丑数数组，但我们发现，有两个队列头都为6，所以我们弹出两个队列头，同时将12,18,30放入三个队列；
				……………………
				疑问：
				1.为什么分三个队列？
				丑数数组里的数一定是有序的，因为我们是从丑数数组里的数乘以2,3,5选出的最小数，一定比以前未乘以2,3,5大，同时对于三个队列内部，按先后顺序乘以2,3,5分别放入，所以同一个队列内部也是有序的；
				2.为什么比较三个队列头部最小的数放入丑数数组？
				因为三个队列是有序的，所以取出三个头中最小的，等同于找到了三个队列所有数中最小的。
				实现思路：
				我们没有必要维护三个队列，只需要记录三个指针显示到达哪一步；“|”表示指针,arr表示丑数数组；
				（1）1
				|2
				|3
				|5
				目前指针指向0,0,0，队列头arr[0] * 2 = 2,  arr[0] * 3 = 3,  arr[0] * 5 = 5
				（2）1 2
				2 |4
				|3 6
				|5 10
				目前指针指向1,0,0，队列头arr[1] * 2 = 4,  arr[0] * 3 = 3, arr[0] * 5 = 5
				（3）1 2 3
				2| 4 6
				3 |6 9
				|5 10 15
				目前指针指向1,1,0，队列头arr[1] * 2 = 4,  arr[1] * 3 = 6, arr[0] * 5 = 5
				
				
				………………
				链接：https://www.nowcoder.com/questionTerminal/6aa9e04fc3794f68acf8778237ba065b
				来源：牛客网
				
				class Solution {
				public:
				    int GetUglyNumber_Solution(int index) {
				        // 0-6的丑数分别为0-6
				        if(index < 7) return index;
				        //p2，p3，p5分别为三个队列的指针，newNum为从队列头选出来的最小数
				        int p2 = 0, p3 = 0, p5 = 0, newNum = 1;
				        vector<int> arr;
				        arr.push_back(newNum);
				        while(arr.size() < index) {
				            //选出三个队列头最小的数
				            newNum = min(arr[p2] * 2, min(arr[p3] * 3, arr[p5] * 5));
				            //这三个if有可能进入一个或者多个，进入多个是三个队列头最小的数有多个的情况
				            if(arr[p2] * 2 == newNum) p2++;
				            if(arr[p3] * 3 == newNum) p3++;
				            if(arr[p5] * 5 == newNum) p5++;
				            arr.push_back(newNum);
				        }
				        return newNum;
				    }
				};
		  */
		

			 public int GetUglyNumber_Solution2(int n)
			     {
			         if(n<=0)return 0;
			         ArrayList<Integer> list=new ArrayList<Integer>();
			         list.add(1);
			         int i2=0,i3=0,i5=0;
			         while(list.size()<n)//循环的条件
			         {
			             int m2=list.get(i2)*2;
			             int m3=list.get(i3)*3;
			             int m5=list.get(i5)*5;
			             int min=Math.min(m2,Math.min(m3,m5));
			             list.add(min);
			             if(min==m2)i2++;
			             if(min==m3)i3++;
			             if(min==m5)i5++;
			         }
			         return list.get(list.size()-1);
			     }
//34.第一个只出现一次的字符；
			 public int FirstNotRepeatingChar(String str) {
			        HashMap<Character,Integer> map = new HashMap<>();
			        for(int i=0;i<str.length();i++){
			            
			            map.put(str.charAt(i),(map.get(str.charAt(i))==null)?1:map.get(str.charAt(i))+1);//不加判断会出现空指针异常
			        }
			        for(int i=0;i<str.length();i++){//string 的length()方法
			            if (map.get(str.charAt(i))==1)
			                return i;
			        }
			        return -1;
			    }
			 
//35数组中的逆序对 归并排序的思想
//在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007
			 public int InversePairs(int [] array) {
			        if(array==null || array.length==0) return 0;
			        int[] newArray = new int[array.length];
			        
			        return InversePairs(array,newArray,0,array.length-1);
			    }
			    
			    
			    public int InversePairs(int[] array,int[] newArray, int low, int high){
			        if(low>=high) return 0;
			        int len = high;//注意这个len的写法，是下表high
			        int mid = (low+high)>>1;
			        int left = InversePairs(array,newArray,low,mid)%1000000007;
			        int right = InversePairs(array,newArray,mid+1,high)%1000000007;
			        int i = mid;
			        int j = high;
			        int count = 0;//不能写成静态变量。因为每一个递归返回的是该次递归的次数。
			        while(i>=low && j>mid){
			            if(array[i] > array[j]){
			                if(count>=1000000007) count%=1000000007;
			                count += (j-mid);
			                newArray[len--] = array[i--];
			            }else{
			                newArray[len--] = array[j--];
			            }
			        }
			        while(i>=low){
			            newArray[len--] = array[i];//不能写成array[i--],因为上面循环结束后i已经--了。
			            i--;
			        } 
			        while(j>mid){
			            newArray[len--] = array[j];
			            j--;
			        } 
			        for(int p=high;p>=low;p--){
			            array[p] = newArray[p];
			        }
			        return (left+right+count)%1000000007;
			    }
//36.两个链表的第一个公共结点
			    //方法一，用空间换时间，用两栈
			    /*public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
			        if(pHead1==null || pHead2==null) return null;
			        Stack<ListNode> s1 = new Stack<>();
			        Stack<ListNode> s2 = new Stack<>();
			        ListNode node = null;
			        while(pHead1!=null){
			            s1.push(pHead1);
			            pHead1 = pHead1.next;
			        }
			        while(pHead2!=null){
			            s2.push(pHead2);
			            pHead2 = pHead2.next;
			        }
			        while(!s1.isEmpty()&&!s2.isEmpty()&&s1.peek() == s2.peek()){//加上这个判断条件!s1.isEmpty()&&!s2.isEmpty()
			            node = s1.pop();
			            s2.pop();
			        }
			        return node;
			    }*/
			    
			    //这个方法不需要空间换时间。
			    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
			        if(pHead1==null || pHead2==null) return null;
			        ListNode current1 = pHead1;
			        ListNode current2 = pHead2;//注意要copy一份，否则计算长度的时候把之前的给弄乱了
			        int p1_len=0;
			        int p2_len=0;
			        while(current1!=null){
			            p1_len++;
			            current1 = current1.next;
			        }
			        while(current2!=null){
			            p2_len++;
			            current2 = current2.next;
			        }
			        int distance = Math.abs(p1_len-p2_len);
			        if(p1_len>p2_len){
			            while(distance!=0){
			                pHead1 = pHead1.next;
			                distance--;
			            }
			        }else if(p1_len<p2_len){
			           while(distance!=0){
			            pHead2 = pHead2.next;
			            distance--;
			           }
			        }
			        while(pHead2!=pHead1){
			                
			                pHead1 = pHead1.next;
			                pHead2 = pHead2.next;
			                
			            }
			        return pHead1;
			    }
			    
 //37.统计一个数字在排序数组中出现的次数。
				
				////因为data中都是整数，所以可以稍微变一下，不是搜索k的两个位置，而是搜索k-0.5和k+0.5
			    //这两个数应该插入的位置，然后相减即可。
			    public int GetNumberOfK(int [] array , int k) {
			       return binSearch(array,k+0.5) - binSearch(array,k-0.5);
			    }
			    private int binSearch(int[] array, double k){
			        int len = array.length;
			        if(len==0 || array==null) return 0;
			        int start = 0;
			        int end = len-1;
			        
			        while(start<=end){
			            int mid = (start+end)>>1;
			            if(array[mid]>k) end = mid-1;
			            else if(array[mid]<k) start = mid+1;

			        }
			        return end;
			    }
			    /*
			     * 链接：https://www.nowcoder.com/questionTerminal/70610bf967994b22bb1c26f9ae901fa2
			来源：牛客网

			public class Solution {
			    public int GetNumberOfK(int [] array , int k) {
			        int length = array.length;
			        if(length == 0){
			            return 0;
			        }
			        int firstK = getFirstK(array, k, 0, length-1);
			        int lastK = getLastK(array, k, 0, length-1);
			        if(firstK != -1 && lastK != -1){
			             return lastK - firstK + 1;
			        }
			        return 0;
			    }
			    //递归写法
			    private int getFirstK(int [] array , int k, int start, int end){
			        if(start > end){
			            return -1;
			        }
			        int mid = (start + end) >> 1;
			        if(array[mid] > k){
			            return getFirstK(array, k, start, mid-1);
			        }else if (array[mid] < k){
			            return getFirstK(array, k, mid+1, end);
			        }else if(mid-1 >=0 && array[mid-1] == k){
			            return getFirstK(array, k, start, mid-1);
			        }else{
			            return mid;
			        }
			    }
			    //循环写法
			    private int getLastK(int [] array , int k, int start, int end){
			        int length = array.length;
			        int mid = (start + end) >> 1;
			        while(start <= end){
			            if(array[mid] > k){
			                end = mid-1;
			            }else if(array[mid] < k){
			                start = mid+1;
			            }else if(mid+1 < length && array[mid+1] == k){
			                start = mid+1;
			            }else{
			                return mid;
			            }
			            mid = (start + end) >> 1;
			        }
			        return -1;
			    }
			}
			     */
			    
//39.二叉树的深度 递归方法
			    //输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。
			    public int TreeDepth(TreeNode root) {
			        if(root==null) return 0;
			        int left = TreeDepth(root.left);
			        int right = TreeDepth(root.right);
			        return (left>right)?(left+1):(right+1);
			    }
			    /*非递归解法 层序遍历
			     * 链接：https://www.nowcoder.com/questionTerminal/435fb86331474282a3499955f0a41e8b
来源：牛客网

 public int TreeDepth(TreeNode pRoot)
    {
        if(pRoot == null){
            return 0;
        }
        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        queue.add(pRoot);
        int depth = 0, count = 0, nextCount = 1;
        while(queue.size()!=0){
            TreeNode top = queue.poll();
            count++;
            if(top.left != null){
                queue.add(top.left);
            }
            if(top.right != null){
                queue.add(top.right);
            }
            if(count == nextCount){
                nextCount = queue.size();
                count = 0;
                depth++;
            }
        }
        return depth;
    }
			     */
//40.判断一棵树是不是平二叉树 性质：它是一 棵空树或它的左右两个子树的高度差的绝对值不超过1，并且左右两个子树都是一棵平衡二叉树。
			    /*public boolean IsBalanced_Solution(TreeNode root) {//结点被重复遍历，不够高效；
			        if(root==null) return true;
			        int left = TreeDepth(root.left);
			        int right = TreeDepth(root.right);
			        int diff = left - right;
			        if(Math.abs(diff)>1) return false;
			        return  IsBalanced_Solution(root.right) && IsBalanced_Solution(root.left);
			    }
			    public int TreeDepth(TreeNode root) {
			        if(root==null) return 0;
			        int left = TreeDepth(root.left);
			        int right = TreeDepth(root.right);
			        return (left>right)?(left+1):(right+1);
			    }*/
			    
			   // 链接：https://www.nowcoder.com/questionTerminal/8b3b95850edb4115918ecebdf1b4d222
			    	//来源：牛客网

			    	//后续遍历时，遍历到一个节点，其左右子树已经遍历  依次自底向上判断，每个节点只需要遍历一次
			    	     
			    	    private boolean isBalanced=true;
			    	    public boolean IsBalanced_Solution(TreeNode root) {
			    	         
			    	        getDepth(root);
			    	        return isBalanced;
			    	    }
			    	    public int getDepth(TreeNode root){
			    	        if(root==null)
			    	            return 0;
			    	        int left=getDepth(root.left);
			    	        int right=getDepth(root.right);
			    	         
			    	        if(Math.abs(left-right)>1){
			    	            isBalanced=false;//未剪枝 把结点全部遍历一遍
			    	        }
			    	        return right>left ?right+1:left+1;
			    	         
			    	    }
			    	    //剪枝了
			    	    /*链接：https://www.nowcoder.com/questionTerminal/8b3b95850edb4115918ecebdf1b4d222
			    	    	来源：牛客网

			    	    	public boolean IsBalanced_Solution(TreeNode root) {
			    	    	        return getDepth(root) != -1;
			    	    	    }
			    	    	     
			    	    	    private int getDepth(TreeNode root) {
			    	    	        if (root == null) return 0;
			    	    	        int left = getDepth(root.left);
			    	    	        if (left == -1) return -1;
			    	    	        int right = getDepth(root.right);
			    	    	        if (right == -1) return -1;
			    	    	        return Math.abs(left - right) > 1 ? -1 : 1 + Math.max(left, right);
			    	    	    }*/
			    	    //用异常达到剪枝效果
			    	    /*
			    	     * 链接：https://www.nowcoder.com/questionTerminal/8b3b95850edb4115918ecebdf1b4d222?toCommentId=509797
							来源：牛客网

					class Solution {
					public:
					    bool IsBalanced_Solution(TreeNode* pRoot) {
					        try{
					            height(pRoot);
					            return true;
					        }catch(string * e){
					            return false;
					        }
					    }
					     
					    int height(TreeNode * root){
					        if(!root)return 0;
					        int left = 1 + height(root->left);
					        int right = 1 + height(root->right);
					        if(abs(left - right)>1)throw new string();
					        return max(left,right);
					    }
					};
								    	     */
//39. 数组中只出现一次的数字 一个整型数组里除了两个数字之外，其他的数字都出现了偶数次。请写程序找出这两个只出现一次的数字。
			    	    /*
								    	     * 链接：https://www.nowcoder.com/questionTerminal/e02fdb54d7524710a7d664d082bb7811
					来源：牛客网
					
					
					 首先我们考虑这个问题的一个简单版本：一个数组里除了一个数字之外，其他的数字都出现了两次。请写程序找出这个只出现一次的数字。
					 这个题目的突破口在哪里？题目为什么要强调有一个数字出现一次，其他的出现两次？我们想到了异或运算的性质：任何一个数字异或它自己都等于0 。也就是说，如果我们从头到尾依次异或数组中的每一个数字，那么最终的结果刚好是那个只出现一次的数字，因为那些出现两次的数字全部在异或中抵消掉了。
					 有了上面简单问题的解决方案之后，我们回到原始的问题。如果能够把原数组分为两个子数组。在每个子数组中，包含一个只出现一次的数字，而其它数字都出现两次。如果能够这样拆分原数组，按照前面的办法就是分别求出这两个只出现一次的数字了。
					 我们还是从头到尾依次异或数组中的每一个数字，那么最终得到的结果就是两个只出现一次的数字的异或结果。因为其它数字都出现了两次，在异或中全部抵消掉了。由于这两个数字肯定不一样，那么这个异或结果肯定不为0 ，也就是说在这个结果数字的二进制表示中至少就有一位为1 。我们在结果数字中找到第一个为1 的位的位置，记为第N 位。现在我们以第N 位是不是1 为标准把原数组中的数字分成两个子数组，第一个子数组中每个数字的第N 位都为1 ，而第二个子数组的每个数字的第N 位都为0 。
					 现在我们已经把原数组分成了两个子数组，每个子数组都包含一个只出现一次的数字，而其它数字都出现了两次。因此到此为止，所有的问题我们都已经解决。
			    	     * 
			    	     */
			    	    public void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {
			    	        int num = array[0];
			    	        for(int i=1;i<array.length;i++){
			    	            num = num ^ array[i];//异或找出两个不同数字的异或
			    	        }
			    	        //找出num的最低位为1的数；
			    	        int point = num^(num-1)&num;
			    	        for(int i=0;i<array.length;i++){
			    	            if((array[i]&point)==0){
			    	                if(num1==null) //这里不要写成 num1[0]==null
			    	                    num1[0]=array[i];
			    	                else
			    	                    num1[0]^=array[i];
			    	            }
			    	            else{
			    	                if(num2==null) 
			    	                    num2[0]=array[i];
			    	                else
			    	                    num2[0]^=array[i];
			    	            }
			    	                
			    	        }
			    	        
			    	        }
	//40.和为S的连续正数序列  输出所有和为S的连续正数序列。序列内按照从小至大的顺序，序列间按照开始数字从小到大的顺序
			    	   

			    	    	public ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {
			    	    	        //存放结果
			    	    	        ArrayList<ArrayList<Integer> > result = new ArrayList<>();
			    	    	        //两个起点，相当于动态窗口的两边，根据其窗口内的值的和来确定窗口的位置和大小
			    	    	        int plow = 1,phigh = 2;
			    	    	        while(phigh > plow){
			    	    	            //由于是连续的，差为1的一个序列，那么求和公式是(a0+an)*n/2
			    	    	            int cur = (phigh + plow) * (phigh - plow + 1) / 2;
			    	    	            //相等，那么就将窗口范围的所有数添加进结果集
			    	    	            if(cur == sum){
			    	    	                ArrayList<Integer> list = new ArrayList<>();
			    	    	                for(int i=plow;i<=phigh;i++){
			    	    	                    list.add(i);
			    	    	                }
			    	    	                result.add(list);
			    	    	                plow++;
			    	    	            //如果当前窗口内的值之和小于sum，那么右边窗口右移一下
			    	    	            }else if(cur < sum){
			    	    	                phigh++;
			    	    	            }else{
			    	    	            //如果当前窗口内的值之和大于sum，那么左边窗口右移一下
			    	    	                plow++;
			    	    	            }
			    	    	        }
			    	    	        return result;
			    	    	    }
		//41 和为S的两个数字  输入一个递增排序的数组和一个数字S，在数组中查找两个数，使得他们的和正好是S，如果有多对数字的和等于S，输出两个数的乘积最小的。
			    	    	public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
			    	            ArrayList<Integer> list = new ArrayList<>();
			    	            if(array==null || array.length<2) return list; //不存在则返回空集合
			    	            int point=0;
			    	             
			    	            for(int i=0;i<array.length;i++){
			    	                if(array[i]>=sum/2){
			    	                    point = i;
			    	                    break;
			    	                }
			    	            }
			    	            for(int i=0;i<point;i++){
			    	                for(int j=array.length-1;j>=point;j--){
			    	                    if((array[i]+array[j]) == sum){
			    	                        list.add(array[i]);
			    	                        list.add(array[j]);
			    	                        return list;
			    	                    }
			    	                }
			    	            }
			    	            return list;
			    	        }
			    	    	/*
			    	    	 * 链接：https://www.nowcoder.com/questionTerminal/390da4f7a00f44bea7c2f3d19491311b
来源：牛客网
既然是排序好的，就好办了：左右加逼

public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {
        ArrayList<Integer> list = new ArrayList<Integer>();
        if (array == null || array.length < 2) {
            return list;
        }
        int i=0,j=array.length-1;
        while(i<j){
            if(array[i]+array[j]==sum){
            list.add(array[i]);
            list.add(array[j]);
                return list;
           }else if(array[i]+array[j]>sum){
                j--;
            }else{
                i++;
            }
             
        }
        return list;
    }
			    	    	 */
					
}
