import input_test
import numpy
import numbapro
from numbapro import cuda
from numba import *
__author__ = 'zhai_jy'


def get_attr(attr_input, attr_num, attr_total, lists, key, offset):# get attr in bitmap_list
	bin32 = 0xffffffff
	bin31 = 0x80000000
	bitmap_list = [[]for i in range(attr_num)]
	lengt = []
	lie = []
	# print "please input the attributes you choose:(if not -1)"
	for i in range(attr_num):
		attrt = attr_input[i]
		if attrt != -1 :
			lengt.append(key[i][attrt])
			lie.append(offset[i][attrt])
		else:
			lengt.append(0)
			lie.append(0)

	for i in range(attr_num):
		attrt = attr_input[i]
		if attrt != -1:
			for j in range(lengt[i]):
				attr_bit = lists[i][lie[i]+j]
				if attr_bit > bin31:
					bitmap_list[i].append(attr_bit)
				else:
					for k in range(attr_bit):
						bitmap_list[i].append(0)

			if len(bitmap_list[i]) < attr_total:
				for j in range(len(bitmap_list[i]), attr_total):
					bitmap_list[i].append(0)

		else:
			for j in range(attr_total): # if All, set 1-Fill
				bitmap_list[i].append(bin32)
	for i in range(attr_num):
		print 'key%d:%d\noffset%d:%d\n' % (i, lengt[i], i, lie[i])
	for bitmap in bitmap_list:
		for j in range(len(bitmap)):
			print "%d:%s" % (j, bin(bitmap[j]).split('b')[1])
		print

	return bitmap_list


@cuda.jit('void(int32[:,:], int32[:], int32, int32, int32)', target = 'gpu')
def index_gpu(bitmap_list, index_list, attr_num, attr_total, attr_mul):# list[] & list[]
	idx = cuda.grid(1)
	for i in range(attr_mul):
		idy = idx * attr_mul + i
		if idy < attr_total:
			num = 0xffffffff
			bin1 = 0x80000000
			for j in range(attr_num):
				num = num & bitmap_list[j][idy]

			num = num << 1
			for j in range(31):
				if num & bin1 == bin1:		#find address
					addr = idy * 31 + j + 1
					index_list[addr-1] = addr

				num = num << 1              



if __name__ == '__main__':
	path1 = 'data_map.pkl'  # get data_map
	path2 = 'bitmap_pic.pkl'   # get bitmap
	attr_input, attr_num, attr_total, lists, key, offset = input_test.input_attr(path1, path2)
	# attr_input is a list that stores the numbers of input attributes 
	# attr_num is the total number of attributes
	# attr_total is the total number of data/31
	if len(attr_input) != attr_num: # there might be a wrong input in input_test.py
		print 'No eligible projects'
	else:
		threadnum = 1024
		blocknum = 1
		attr_mul = (attr_total + (threadnum * blocknum - 1))/(threadnum * blocknum)
		# attr_mul is the number that each thread need to be performed
		print '---index----\nattr_num:%d\nattr_total:%d\nattr_mul:%d\n----------' % (attr_num, attr_total, attr_mul)
		attr_num = 1
		print '---change_num----\nattr_num:%d\nattr_total:%d\nattr_mul:%d\n----------' % (attr_num, attr_total, attr_mul)
		index_list = numpy.zeros(attr_total*31, dtype = 'int32')
		bitmap_list = get_attr(attr_input, attr_num, attr_total, lists, key, offset)
		stream = cuda.stream()
		d_bitmap_list = cuda.to_device(numpy.array(bitmap_list), stream)
		d_index_list = cuda.to_device(numpy.array(index_list), stream)
		index_gpu[blocknum, threadnum, stream](d_bitmap_list, d_index_list, attr_num, attr_total, attr_mul)
		index_list = d_index_list.copy_to_host()
		stream.synchronize()
		print '----index_list------'
		print '[',
		for index in index_list:
			if index != 0:
				print '%d,' % index,
		print ']'
			



