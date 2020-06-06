import time
t = time.localtime()

def Acheck_TR(anum,atime,person_list,frame,total_time):
  for i in range( len( anum ) ) :
    atime[ i ]=round( anum[ i ] / (frame/total_time) , 1 )
  for i in range( len( atime ) ) :
    person_list[ i ].appear=atime[ i ]
  for i in range( len( anum ) ) :
    person_list[ i ].percent=round( anum[i]/frame * 100 , 1 )
    if person_list[i].percent > 100:
      person_list[i].percent = 100
  s='{}'.format( (lambda x : x if x >= 10 else '0' + '{}'.format( x ))( t.tm_mon ) + '/' ) + '{}'.format(
    (lambda x : x if x >= 10 else '0' + '{}'.format( x ))( t.tm_mday ) ) + ' ' + '{}'.format(
    t.tm_hour ) + ':' + '{}'.format( t.tm_min ) + ':' + '{}'.format( t.tm_sec )
  print('\033[33m'+'[{} 출석체크결과] :'.format( s ))

  total_time_r = round(total_time,1)
  if total_time >=60 :
    total_time_r = "{}분 {}초".format(total_time_r//60,total_time_r%60)
  else :
    total_time_r = "{}초".format(total_time_r)
  print('\033[96m'+"총 시간: "+total_time_r+ '\033[0m')

  for i in range(len(atime)):
    if person_list[i].result == '지각(late)' or person_list[i].result == '결석(absence)':
      continue
    elif person_list[i].percent < 60:
      person_list[ i ].result= '\033[31m' + '출튀(escape)'+ '\033[0m'
    elif person_list[i].percent >=60:
      person_list[ i ].result='출석(attend)'



  print( '\033[32m'+'|ㅡㅡ학번ㅡㅡ|ㅡㅡ전공ㅡㅡ|ㅡㅡ이름ㅡㅡ|ㅡ출석시간ㅡ|ㅡㅡ빈도ㅡㅡ|ㅡㅡ결과ㅡㅡ|' '\033[0m')
  for i in range( len( person_list ) ) :
    print( '  {}'.format( person_list[ i ].SN ) + '   {}'.format( person_list[ i ].major ) + '    {}'.format(
      person_list[ i ].name ) + '      {}초'.format( person_list[ i ].appear ) + '       {}%'.format(
      person_list[ i ].percent )+'    {}'.format(
person_list[ i ].result ) )

