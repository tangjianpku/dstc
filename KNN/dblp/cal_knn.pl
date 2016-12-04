#! /usr/bin/perl

$infile=$ARGV[0];
$outfile=$ARGV[1];
$train_num=$ARGV[2];
$neighbors=$ARGV[3];

open(IN, $infile) or die("cannot open $infile");
open(OUT, ">$outfile") or die("cannot open $outfile");
$id=0;
@arrays=();
while($line=<IN>)
{
	chomp $line;
	if($id==0)
	{
		@cols =split(' ', $line);
		$doc_num =$cols[0];
		$dim =$cols[1];
	}else
	{
		@cols =split(' ', $line);
		$len  =scalar @cols;
		my @array=();
		for($i=1;$i<$len;$i++)
		{
		   push @array, $cols[$i];
		}
		push @arrays, [@array];
	}
	$id++;	
}

$doc_num =scalar @arrays;

print $doc_num."\n";

#foreach $ele (@arrays)
#{
#	@array=@{$ele};
#	$len =scalar @array;
	#print $len."\n";
#	foreach $ele(@array)
#	{	
#		print $ele." ";
	#}
	#print "\n";
	#last;
#}
for($i=0;$i<$doc_num; $i++)
{	@fea1=@{$arrays[$i]};
	$sum1=0;
	for($d=0;$d<$dim;$d++)
	{
		$sum1+=$fea1[$d]*$fea1[$d];
	}
	my %sim={};
	for($j=0;$j<$train_num;$j++)
	{
		@fea2=@{$arrays[$j]};
		$sum=0;
		$sum2=0;
		for($d=0;$d<$dim;$d++)
		{
			#print $fea2[$d]." ";
			$sum+=$fea1[$d]*$fea2[$d];
			$sum2+=$fea2[$d]*$fea2[$d];
		}
		#print "\n...";
		if($j==$i)
		{
			$sum=0;
		}
		$sim{$j}=$sum/sqrt($sum1)/sqrt($sum2);
	}
	### sorting here
	$count=0;
	print "DOC iD:".$i."\n";;
	foreach my $key (sort {$sim{$b}<=> $sim{$a}} keys %sim)
	{
		print OUT $key." ";
		$count++;
		if($count==$neighbors)
		{
			last;
		}	
	}
	print OUT "\n";	
}
