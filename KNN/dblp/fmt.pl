#! /usr/bin/perl

$infile=$ARGV[0];
$outfile=$ARGV[1];
$size=$ARGV[2];
$neighbor_size=$ARGV[3];
open(IN, $infile) or die("cannot oepn $infile");
open(OUT, ">$outfile") or die("cannot open $outfile");

while($line=<IN>)
{
	chomp $line;
	@cols=split(' ',$line);
	if($cols[1]<$size)
	{
		$knn{$cols[0]}{$cols[1]}=$cols[2];
	}
}
foreach my $key (sort{$a<=>$b}keys %knn)
{
	%subhash =%{$knn{$key}};
	$count=0;
	#print OUT $key;
	foreach my $key2 (sort{$subhash{$b}<=>$subhash{$a}} keys %subhash)
	{	
		if($count==0)
		{
			print OUT $key2;
		}else
		{
			print OUT " ".$key2;
		}
		$count++;
		if($count==$neighbor_size)
		{
			last;
		}
	
	}
	print OUT "\n";
}
close(IN);
close(OUT);
